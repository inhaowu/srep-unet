import argparse
import concurrent.futures
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import vtk
try:
    from vtkmodules.util import numpy_support as vtk_np
except ImportError:
    from vtk.util import numpy_support as vtk_np
from scipy.sparse.linalg import spsolve


def _resolve_slicer_libs_dir() -> Path:
    env_path = os.environ.get("SLICER_LIBS_DIR")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    script_dir = Path(__file__).resolve().parent
    candidates.extend([
        script_dir / "lib" / "SlicerSALT-5.9" / "qt-scripted-modules" / "Resources" / "Libraries",
        script_dir / "share" / "SlicerSALT-5.9" / "qt-scripted-modules" / "Resources" / "Libraries",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Cannot find SlicerSALT Libraries directory. Set SLICER_LIBS_DIR explicitly."
    )


_LIBS = _resolve_slicer_libs_dir()
sys.path.insert(0, str(_LIBS))
sys.path.insert(0, str(_LIBS / "shanapy_private"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

lapy = None
Srep = None
get_perfect_ellipsoid_and_srep_from_quasi = None


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is required but not installed in this Python environment. "
            "Install it in Slicer Python, or run with an interpreter that has pyvista."
        ) from exc
    return pv


def _require_pipeline_deps():
    global lapy, Srep, get_perfect_ellipsoid_and_srep_from_quasi
    if lapy is not None and Srep is not None and get_perfect_ellipsoid_and_srep_from_quasi is not None:
        return
    try:
        import lapy as _lapy
    except ImportError as exc:
        raise RuntimeError(
            "lapy is required but not installed in this Python environment."
        ) from exc
    try:
        from nick_srep import Srep as _Srep
        from shanapy_private.fitting_srep_to_quasi import (
            get_perfect_ellipsoid_and_srep_from_quasi as _fit_srep,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import shanapy modules from '{_LIBS}'. Set SLICER_LIBS_DIR if needed."
        ) from exc

    lapy = _lapy
    Srep = _Srep
    get_perfect_ellipsoid_and_srep_from_quasi = _fit_srep


def _preflight_check(deformetrica_cmd: str):
    _require_pyvista()
    _require_pipeline_deps()
    if shutil.which(deformetrica_cmd) is None and not Path(deformetrica_cmd).exists():
        raise RuntimeError(
            f"deformetrica executable not found: '{deformetrica_cmd}'. "
            "Use --deformetrica-cmd /full/path/to/deformetrica"
        )


def _clean_subprocess_env():
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONEXECUTABLE", None)
    return env


def vtk_to_lapy(pd):
    _require_pipeline_deps()
    points = vtk_np.vtk_to_numpy(pd.GetPoints().GetData())
    polys = vtk_np.vtk_to_numpy(pd.GetPolys().GetData())
    polys = polys.reshape([polys.size // 4, 4])[:, 1:]
    return lapy.TriaMesh(points, polys)


def lapy_to_vtk(mesh):
    npd = vtk.vtkPolyData()
    npts = vtk.vtkPoints()
    npts.SetData(vtk_np.numpy_to_vtk(mesh.v))
    npd.SetPoints(npts)
    polys = 3 * np.ones([mesh.t.shape[0], 4], dtype=np.int64)
    polys[:, 1:] = mesh.t
    npolys = vtk.vtkCellArray()
    npolys.SetCells(polys.shape[0], vtk_np.numpy_to_vtkIdTypeArray(polys, deep=True))
    npd.SetPolys(npolys)
    return npd


def tria_mean_curvature_flow(tria, max_iter=40, stop_eps=1e-12, step=0.001):
    _require_pipeline_deps()
    meshes = []
    trianorm = lapy.TriaMesh(tria.v, tria.t)
    trianorm.normalize_()
    meshes.append(lapy.TriaMesh(trianorm.v, trianorm.t))

    fem = lapy.Solver(trianorm, lump=True)
    a_mat = fem.stiffness

    for x in range(max_iter):
        vlast = trianorm.v
        mass = lapy.Solver.fem_tria_mass(trianorm, lump=True)
        mass_v = mass.dot(trianorm.v)
        trianorm.v = spsolve(mass + step * a_mat, mass_v)
        trianorm.normalize_()
        meshes.append(lapy.TriaMesh(trianorm.v, trianorm.t))
        dv = trianorm.v - vlast
        diff = np.trace(np.square(np.matmul(dv.T, mass.dot(dv))))
        if diff < stop_eps:
            logging.info("  cMCF converged at iter %d", x + 1)
            break
    return trianorm, meshes


def run_backward_stage(start_file: Path, stop_file: Path, reg_dir: Path, deformetrica_cmd: str):
    model_xml = f"""<model>
    <model-type>Registration</model-type>
    <dimension>3</dimension>
    <template>
        <object id="mesh">
            <deformable-object-type>SurfaceMesh</deformable-object-type>
            <attachment-type>Varifold</attachment-type>
            <noise-std>0.01</noise-std>
            <kernel-width>0.5</kernel-width>
            <kernel-type>torch</kernel-type>
            <filename>{start_file.resolve()}</filename>
        </object>
    </template>
    <deformation-parameters>
        <kernel-width>0.1</kernel-width>
        <kernel-type>torch</kernel-type>
        <number-of-timepoints>10</number-of-timepoints>
    </deformation-parameters>
</model>"""
    (reg_dir / "model.xml").write_text(model_xml)

    dataset_xml = f"""<data-set>
    <subject id="subj1">
        <visit id="experiment">
            <filename object_id="mesh">{stop_file.resolve()}</filename>
        </visit>
    </subject>
</data-set>"""
    (reg_dir / "data_set.xml").write_text(dataset_xml)

    opt_xml = f"""<?xml version="1.0"?>
<optimization-parameters>
    <optimization-method-type>ScipyLBFGS</optimization-method-type>
    <gpu-mode>{os.environ.get("DEFORMETRICA_GPU_MODE", "full")}</gpu-mode>
</optimization-parameters>"""
    (reg_dir / "optimization_parameters.xml").write_text(opt_xml)

    subprocess.run(
        [
            deformetrica_cmd,
            "estimate",
            "model.xml",
            "data_set.xml",
            "-p",
            "optimization_parameters.xml",
        ],
        cwd=reg_dir,
        env=_clean_subprocess_env(),
        check=True,
    )


def run_shooting(reg_path: Path, shooting_path: Path, srep_file: Path, deformetrica_cmd: str):
    _require_pipeline_deps()
    pv = _require_pyvista()
    reg_out_dir = reg_path / "output"
    model_xml = f"""<model>
    <model-type>Shooting</model-type>
    <dimension>3</dimension>
    <template>
        <object id="srep">
            <deformable-object-type>PolyLine</deformable-object-type>
            <filename>{srep_file.resolve()}</filename>
        </object>
    </template>
    <initial-control-points>{(reg_out_dir / "DeterministicAtlas__EstimatedParameters__ControlPoints.txt").resolve()}</initial-control-points>
    <initial-momenta>{(reg_out_dir / "DeterministicAtlas__EstimatedParameters__Momenta.txt").resolve()}</initial-momenta>
    <deformation-parameters>
        <kernel-width>0.1</kernel-width>
        <kernel-type>torch</kernel-type>
        <number-of-timepoints>10</number-of-timepoints>
    </deformation-parameters>
</model>"""
    (shooting_path / "model.xml").write_text(model_xml)

    subprocess.run(
        [deformetrica_cmd, "compute", "model.xml"],
        cwd=shooting_path,
        env=_clean_subprocess_env(),
        check=True,
    )

    srep = Srep()
    srep.load_polydata_file(str(shooting_path / "output" / "Shooting__GeodesicFlow__srep__tp_10__age_1.00.vtk"))
    surf = pv.read(reg_out_dir / "DeterministicAtlas__Reconstruction__mesh__subject_subj1.vtk")
    srep.regularize_crest(0.1)
    srep.regularize_skeleton()
    srep.refine_spoke_lengths(surf)
    pv.wrap(srep.polydata).save(str(shooting_path / "regularized_output_srep.vtk"))


def srep_to_json(srep_vtk, outfile: Path):
    def pt(i):
        return srep_vtk.GetPoints().GetPoint(i)

    def make_spoke(skel_idx, bnd_idx):
        sp = {"CoordinateSystem": "LPS", "Value": list(pt(skel_idx))}
        sd = {
            "CoordinateSystem": "LPS",
            "Value": (np.array(pt(bnd_idx)) - np.array(pt(skel_idx))).tolist(),
        }
        return {"SkeletalPoint": sp, "Direction": sd}

    all_skel = []

    for i in range(0, 26, 2):
        skel = []
        ind = i
        skel.append({"UpSpoke": make_spoke(ind, ind + 1), "DownSpoke": make_spoke(ind + 122, ind + 123)})
        ind += 26
        skel.append({"UpSpoke": make_spoke(ind, ind + 1), "DownSpoke": make_spoke(ind + 122, ind + 123)})
        ind += 48
        skel.append(
            {
                "UpSpoke": make_spoke(ind, ind + 1),
                "DownSpoke": make_spoke(ind + 122, ind + 123),
                "CrestSpoke": make_spoke(ind + 170, ind + 171),
            }
        )
        all_skel.append(skel)

    for i in range(22, 0, -2):
        skel = []
        ind = i
        skel.append({"UpSpoke": make_spoke(ind, ind + 1), "DownSpoke": make_spoke(ind + 122, ind + 123)})
        ind = i + 26 + 24
        skel.append({"UpSpoke": make_spoke(ind, ind + 1), "DownSpoke": make_spoke(ind + 122, ind + 123)})
        ind += 48
        skel.append(
            {
                "UpSpoke": make_spoke(ind, ind + 1),
                "DownSpoke": make_spoke(ind + 122, ind + 123),
                "CrestSpoke": make_spoke(ind + 170, ind + 171),
            }
        )
        all_skel.append(skel)

    result = {"EllipticalSRep": {"CrestPoints": 24, "Steps": 2, "Skeleton": all_skel}}
    outfile.write_text(json.dumps(result, indent=2))
    logging.info("S-rep JSON saved: %s", outfile)


def merge_crest_skeletal_inplace(data):
    skel = data["EllipticalSRep"]["Skeleton"]
    for col in skel:
        for layer in col:
            if "CrestSpoke" not in layer:
                continue
            p_ud = np.array(layer["UpSpoke"]["SkeletalPoint"]["Value"], dtype=float)
            p_crest = np.array(layer["CrestSpoke"]["SkeletalPoint"]["Value"], dtype=float)
            d_crest = np.array(layer["CrestSpoke"]["Direction"]["Value"], dtype=float)
            surface_pt = p_crest + d_crest
            d_new = surface_pt - p_ud
            layer["CrestSpoke"]["SkeletalPoint"]["Value"] = p_ud.tolist()
            layer["CrestSpoke"]["Direction"]["Value"] = d_new.tolist()


def export_srep_json_to_vtk(input_json: Path, out_dir: Path, prefix: str):
    pv = _require_pyvista()
    with input_json.open("r") as f:
        data = json.load(f)

    skel_data = data["EllipticalSRep"]["Skeleton"]
    all_pts = []
    pt_index = {}

    def get_or_add(pt):
        key = tuple(float(x) for x in pt)
        if key not in pt_index:
            pt_index[key] = len(all_pts)
            all_pts.append(key)
        return pt_index[key]

    spokes = []
    type_map = {"UpSpoke": 0, "DownSpoke": 1, "CrestSpoke": 2}

    for col in skel_data:
        for layer in col:
            for spoke_type in ["UpSpoke", "DownSpoke", "CrestSpoke"]:
                if spoke_type not in layer:
                    continue
                sp = layer[spoke_type]["SkeletalPoint"]["Value"]
                d = layer[spoke_type]["Direction"]["Value"]
                bp = [sp[i] + d[i] for i in range(3)]
                i_skel = get_or_add(sp)
                i_bnd = get_or_add(bp)
                spokes.append((i_skel, i_bnd, spoke_type))

    pts_arr = np.array(all_pts, dtype=np.float32)


    lines_flat = []
    spoke_type_ids = []
    for (i_s, i_b, stype) in spokes:
        lines_flat.extend([2, i_s, i_b])
        spoke_type_ids.append(type_map[stype])

    mesh_spokes = pv.PolyData()
    mesh_spokes.points = pts_arr
    mesh_spokes.lines = np.array(lines_flat, dtype=np.int32)
    mesh_spokes.cell_data["spoke_type"] = np.array(spoke_type_ids, dtype=np.int32)
    out_spokes = out_dir / f"{prefix}_spokes.vtk"
    mesh_spokes.save(out_spokes)


    grid_idx = np.zeros((24, 3), dtype=np.int32)
    for ci, col in enumerate(skel_data):
        for li, layer in enumerate(col):
            sp = layer["UpSpoke"]["SkeletalPoint"]["Value"]
            grid_idx[ci, li] = get_or_add(sp)

    quads = []
    for ci in range(23):
        for li in range(2):
            quads.append([grid_idx[ci, li], grid_idx[ci + 1, li], grid_idx[ci + 1, li + 1], grid_idx[ci, li + 1]])
    for li in range(2):
        quads.append([grid_idx[23, li], grid_idx[0, li], grid_idx[0, li + 1], grid_idx[23, li + 1]])

    faces_flat = []
    for q in quads:
        faces_flat.extend([4] + q)

    mesh_skel = pv.PolyData()
    mesh_skel.points = pts_arr
    mesh_skel.faces = np.array(faces_flat, dtype=np.int32)
    out_skel = out_dir / f"{prefix}_skeletal.vtk"
    mesh_skel.save(out_skel)


    bnd_pts = []
    bnd_type = []
    for (i_s, i_b, stype) in spokes:
        bnd_pts.append(all_pts[i_b])
        bnd_type.append(type_map[stype])

    mesh_bnd = pv.PolyData(np.array(bnd_pts, dtype=np.float32))
    mesh_bnd["spoke_type"] = np.array(bnd_type, dtype=np.int32)
    out_bnd = out_dir / f"{prefix}_boundary.vtk"
    mesh_bnd.save(out_bnd)

    logging.info("Saved VTK exports: %s, %s, %s", out_spokes.name, out_skel.name, out_bnd.name)


def extract_case_name(vtk_file: Path):
    stem = vtk_file.stem
    suffix = "_SPHARM"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def process_one(input_vtk: Path, output_path: Path, deformetrica_cmd: str, merge_crest: bool, export_vtk: bool):
    _require_pipeline_deps()
    pv = _require_pyvista()
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info("  Read mesh: %s", input_vtk.name)
    r = vtk.vtkPolyDataReader()
    r.SetFileName(str(input_vtk))
    r.Update()
    pd = r.GetOutput()
    if pd.GetNumberOfPoints() == 0:
        r2 = vtk.vtkXMLPolyDataReader()
        r2.SetFileName(str(input_vtk))
        r2.Update()
        pd = r2.GetOutput()
    if pd.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Cannot read mesh: {input_vtk}")
    logging.info("  #points=%d, #cells=%d", pd.GetNumberOfPoints(), pd.GetNumberOfCells())

    mesh = vtk_to_lapy(pd)
    norm_mesh = lapy.TriaMesh(mesh.v, mesh.t)
    norm_mesh.normalize_()
    translation = np.mean(mesh.v, axis=0) - np.mean(norm_mesh.v, axis=0)
    scale = np.linalg.norm(mesh.v - translation) / np.linalg.norm(norm_mesh.v)

    logging.info("  Run cMCF ...")
    flow_mesh, meshes = tria_mean_curvature_flow(norm_mesh, max_iter=40, stop_eps=1e-12, step=0.001)
    mesh_dir = output_path / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    for i, m in enumerate(meshes):
        pv.wrap(lapy_to_vtk(m)).save(str(mesh_dir / f"{i:02d}.vtk"))

    logging.info("  Fit perfect ellipsoid ...")
    new_pd = lapy_to_vtk(flow_mesh)
    pe, srep_init, _, _, _, _ = get_perfect_ellipsoid_and_srep_from_quasi(
        pv.wrap(new_pd), num_spine_points=13, num_skeletal_onionskins=4
    )
    pe.save(str(mesh_dir / "PE.vtk"))

    srep_dir = output_path / "sreps"
    srep_dir.mkdir(exist_ok=True)
    pv.wrap(srep_init).save(str(srep_dir / "PE_srep_regular.vtk"))

    dec = vtk.vtkDecimatePro()
    dec.SetInputData(pe)
    dec.SetTargetReduction(0.9)
    dec.Update()
    pe_down_file = mesh_dir / "PE_downsample.vtk"
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(str(pe_down_file))
    w.SetInputData(dec.GetOutput())
    w.Update()

    logging.info("  Run backward registrations ...")
    reg_dir = output_path / "registrations"
    reg_dir.mkdir(exist_ok=True)
    start_num = len(meshes) - 1
    reg_out_dir = reg_dir / f"PE_{start_num}"
    reg_out_dir.mkdir(exist_ok=True)
    run_backward_stage(pe_down_file, mesh_dir / f"{start_num:02d}.vtk", reg_out_dir, deformetrica_cmd)

    used_nums = ["PE", start_num]
    last_dir = reg_out_dir / "output"
    last_num = start_num
    next_num = math.floor(start_num / 10.0) * 10
    if start_num % 10 == 0:
        next_num = start_num - 5

    while next_num > -1:
        start_file = last_dir / "DeterministicAtlas__Reconstruction__mesh__subject_subj1.vtk"
        stop_file = mesh_dir / f"{next_num:02d}.vtk"
        reg_out_dir = reg_dir / f"{last_num}_{next_num}"
        reg_out_dir.mkdir(exist_ok=True)
        run_backward_stage(start_file, stop_file, reg_out_dir, deformetrica_cmd)
        last_dir = reg_out_dir / "output"
        last_num = next_num
        used_nums.append(last_num)
        if last_num > 10:
            next_num = last_num - 5
        elif last_num == 10:
            next_num = 7
        elif last_num == 7:
            next_num = 4
        else:
            next_num = last_num - 1

    logging.info("  Run geodesic shootings ...")
    shoot_dir = output_path / "shootings"
    shoot_dir.mkdir(exist_ok=True)

    dir_name = f"{used_nums[0]}_{used_nums[1]}"
    shoot_dir_name = shoot_dir / dir_name
    shoot_dir_name.mkdir(exist_ok=True)
    run_shooting(reg_dir / dir_name, shoot_dir_name, srep_dir / "PE_srep_regular.vtk", deformetrica_cmd)

    last_shoot_dir = shoot_dir_name
    for i in range(1, len(used_nums) - 1):
        dir_name = f"{used_nums[i]}_{used_nums[i + 1]}"
        shoot_dir_name = shoot_dir / dir_name
        shoot_dir_name.mkdir(exist_ok=True)
        run_shooting(
            reg_dir / dir_name,
            shoot_dir_name,
            last_shoot_dir / "regularized_output_srep.vtk",
            deformetrica_cmd,
        )
        last_shoot_dir = shoot_dir_name

    logging.info("  Export JSON ...")
    r_final = vtk.vtkPolyDataReader()
    r_final.SetFileName(str(last_shoot_dir / "regularized_output_srep.vtk"))
    r_final.Update()
    srep_final = r_final.GetOutput()

    pts = vtk_np.vtk_to_numpy(srep_final.GetPoints().GetData())
    pts = pts * scale + translation
    srep_final.GetPoints().SetData(vtk_np.numpy_to_vtk(pts))

    out_json = output_path / "final_srep.srep.json"
    srep_to_json(srep_final, out_json)

    merged_json = None
    if merge_crest:
        with out_json.open("r") as f:
            merged_data = json.load(f)
        merge_crest_skeletal_inplace(merged_data)
        merged_json = output_path / "final_srep_merged.srep.json"
        merged_json.write_text(json.dumps(merged_data, indent=2))
        logging.info("Merged crest skeletal points: %s", merged_json)

    if export_vtk:
        src_json = merged_json if merged_json else out_json
        prefix = "final_srep_merged" if merged_json else "final_srep"
        export_srep_json_to_vtk(src_json, output_path, prefix)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evolutionary S-rep fitting for VTK meshes")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input VTK meshes")
    parser.add_argument("--output-base-dir", type=Path, required=True, help="Output base directory")
    parser.add_argument("--vtk-pattern", default="*_SPHARM.vtk", help="Glob pattern for mesh files")
    parser.add_argument("--deformetrica-cmd", default="deformetrica", help="Deformetrica executable or full path")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases with existing final_srep.srep.json")
    parser.add_argument("--merge-crest", action="store_true", help="Also write final_srep_merged.srep.json")
    parser.add_argument("--export-vtk", action="store_true", help="Export spokes/skeletal/boundary VTK from JSON")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent cases to process")
    return parser.parse_args()


def _process_case_worker(
    vtk_file_str: str,
    output_path_str: str,
    deformetrica_cmd: str,
    merge_crest: bool,
    export_vtk: bool,
):
    vtk_file = Path(vtk_file_str)
    output_path = Path(output_path_str)
    case_name = extract_case_name(vtk_file)
    try:
        process_one(
            vtk_file,
            output_path,
            deformetrica_cmd=deformetrica_cmd,
            merge_crest=merge_crest,
            export_vtk=export_vtk,
        )
        return case_name, True, ""
    except Exception as exc:
        return case_name, False, f"{exc}\n{traceback.format_exc()}"


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    _preflight_check(args.deformetrica_cmd)

    input_dir = args.input_dir
    output_base = args.output_base_dir
    output_base.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(input_dir.glob(args.vtk_pattern))
    if not vtk_files:
        logging.error("No mesh matched: %s", input_dir / args.vtk_pattern)
        return

    total = len(vtk_files)
    success = 0
    skipped = 0
    failed = []
    jobs = []

    logging.info("Found %d files. Start batch processing...", total)
    logging.info("=" * 60)

    for idx, vtk_file in enumerate(vtk_files, 1):
        case_name = extract_case_name(vtk_file)
        output_path = output_base / case_name

        logging.info("[%d/%d] %s", idx, total, case_name)

        if args.skip_existing and (output_path / "final_srep.srep.json").exists():
            logging.info("  -> exists, skipped")
            skipped += 1
            continue

        jobs.append((vtk_file, output_path, case_name))

    if not jobs:
        logging.info("No pending cases after skip filtering.")
    elif args.workers == 1:
        for vtk_file, output_path, case_name in jobs:
            try:
                process_one(
                    vtk_file,
                    output_path,
                    deformetrica_cmd=args.deformetrica_cmd,
                    merge_crest=args.merge_crest,
                    export_vtk=args.export_vtk,
                )
                logging.info("%s -> done", case_name)
                success += 1
            except Exception as e:
                logging.error("%s -> failed: %s", case_name, e)
                traceback.print_exc()
                failed.append(case_name)
            logging.info("-" * 60)
    else:
        logging.info("Run pending cases with %d workers...", args.workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _process_case_worker,
                    str(vtk_file),
                    str(output_path),
                    args.deformetrica_cmd,
                    args.merge_crest,
                    args.export_vtk,
                ): case_name
                for vtk_file, output_path, case_name in jobs
            }
            done = 0
            for future in concurrent.futures.as_completed(future_map):
                done += 1
                case_name = future_map[future]
                try:
                    _, ok, err = future.result()
                except Exception as e:
                    ok = False
                    err = f"{e}\n{traceback.format_exc()}"
                if ok:
                    success += 1
                    logging.info("[%d/%d] %s -> done", done, len(jobs), case_name)
                else:
                    failed.append(case_name)
                    logging.error("[%d/%d] %s -> failed\n%s", done, len(jobs), case_name, err)
                logging.info("-" * 60)

    logging.info("=" * 60)
    logging.info("Batch finished")
    logging.info("  success: %d / %d", success, total)
    logging.info("  skipped: %d", skipped)
    logging.info("  failed: %d", len(failed))
    if failed:
        logging.info("  failed cases: %s", failed)


if __name__ == "__main__":
    main()
