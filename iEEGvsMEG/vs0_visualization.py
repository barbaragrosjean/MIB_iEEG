"""Export model feature weights to folded-left MNI volumes and Workbench metrics.

Edit the settings below, then run: python visualization.py
The template NIfTI and both surfaces must share the input MNI millimeter space.
Outputs are projection weights, not activation/forward patterns or significance.
"""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
from nilearn.datasets import load_mni152_template

# Analysis settings
ROOT = Path(__file__).resolve().parent
MODEL = 'plssvd'                    # separate_pca, joint_pca, plssvd
MEG_KIND = 'paired_coverage'       # full_average, coverage_average, paired_coverage, random_control also supported
COMPONENT = 1                    # one-based component index
BLOCK_SCALING = 'equal_variance'              # none or equal_variance (whole-modality scaling)
CONDITION_MODE = 'average'          # average or stack
SEED = 2026

# Data and template paths: replace template placeholders with your actual files.
MEG_DIR = ROOT / 'MEG' / 'dataMEG'
IEEG_DIR = ROOT / 'ieeg_shortWOBS_fs250'
METADATA_CSV = None                 # optional CSV; otherwise existing GetInfo loader
MEG_TIMES_FILE = None
MEG_TMIN = None                     # None uses first iEEG time; verify MEG alignment

# templates
TEMPLATE_NIFTI = ROOT.parent / 'templates' / 'MNI_template.nii.gz'
LEFT_SURFACE = ROOT.parent / 'templates'/ 'ParcellationPilot.L.midthickness.32k_fs_LR.surf.gii'
RIGHT_SURFACE = ROOT.parent / 'templates' / 'ParcellationPilot.R.midthickness.32k_fs_LR.surf.gii'

WB_COMMAND = 'wb_command'          # or absolute path to the executable
RADIUS_MM = 4.                     # nearest-location display radius; no whole-brain extrapolation
OUTPUT_DIR = ROOT / 'out' / 'workbench' / f'{MODEL}_{MEG_KIND}_{BLOCK_SCALING}_pc{COMPONENT:03d}'


def compute_weights(output_dir=OUTPUT_DIR):
    """Load current pipeline datasets, fit models, save selected weights + original MNI coordinates.

    Uses the existing common fitter (which computes all three models), then
    selects MODEL. Separate PCA component numbers are independent across modalities.
    Weights remain in the preprocessed/scaled feature basis, without sign flipping.
    """
    from coverage_matching_utils import load_dataset
    from cov_models_utils import fit_cov_models
    import sys
    for directory in (ROOT.parent, ROOT.parent/'LB'):
        sys.path.insert(0, str(directory))
    if MODEL not in ('separate_pca','joint_pca','plssvd'):
        raise ValueError('Unknown MODEL.')
    if isinstance(COMPONENT,bool) or not isinstance(COMPONENT,int) or COMPONENT<1:
        raise ValueError('COMPONENT must be a positive one-based integer.')
    tmin=MEG_TMIN
    if tmin is None and MEG_TIMES_FILE is None:
        files=sorted(IEEG_DIR.glob('*_info.json'))
        if not files:raise FileNotFoundError(f'No iEEG info files in {IEEG_DIR}')
        tmin=float(json.loads(files[0].read_text())['time_epoch'][0])
    ieeg=load_dataset('ieeg',meg_dir=MEG_DIR,ieeg_dir=IEEG_DIR,
        metadata_csv=METADATA_CSV,ieeg_coordinate_unit='mm',meg_coordinate_unit='m',
        meg_times_file=MEG_TIMES_FILE,meg_tmin=tmin,condition_mode=CONDITION_MODE)
    meg=load_dataset(MEG_KIND,reference=ieeg,seed=SEED)
    model=fit_cov_models(ieeg,meg,n_components=COMPONENT,block_scaling=BLOCK_SCALING)[MODEL]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    files={}
    for modality,dataset in [('ieeg',ieeg),('meg',meg)]:
        weights=getattr(model,modality+'_weights')
        if weights.shape[1]<COMPONENT:raise ValueError('Requested component exceeds data rank.')
        table=dataset.metadata.reset_index(drop=True).copy()
        if len(table)!=len(weights):raise ValueError('Metadata and weights do not align.')
        table['weight']=weights[:,COMPONENT-1]
        files[modality]=out/f'{modality}_weights.csv';table.to_csv(files[modality],index=False)
    (out/'settings.json').write_text(json.dumps(dict(model=MODEL,meg_kind=MEG_KIND,
        component=COMPONENT,block_scaling=BLOCK_SCALING,condition_mode=CONDITION_MODE,
        seed=SEED,modality_scales=list(model.scales),coordinate_unit='mm',
        radius_mm=RADIUS_MM,folding='x -> -abs(x); coincident coordinates averaged'),indent=2))
    return files


def create_nifti(weight_files, template=TEMPLATE_NIFTI, radius_mm=RADIUS_MM):
    """Fold coordinates left; average exact duplicates; paint nearest weights within radius.

    Original CSVs are unchanged. Left contains merged data only at x<0;
    right is an explicitly empty volume. Coverage files distinguish absent
    data from measured zero weights. Signed coincident weights may cancel.
    """
    import nibabel as nib
    import pandas as pd
    from scipy.spatial import cKDTree
    if not np.isfinite(radius_mm) or radius_mm<=0:
        raise ValueError('radius_mm must be positive.')
    ref=nib.load(str(template))
    if len(ref.shape)!=3:raise ValueError('Use a 3D MNI template.')
    if ref.header.get_xyzt_units()[0]!='mm':
        raise ValueError('Template spatial units must be mm.')
    outputs={}
    for modality,path in weight_files.items():
        path=Path(path);table=pd.read_csv(path)
        points=table[['x','y','z','weight']].copy()
        if points.empty or not np.isfinite(points.to_numpy()).all():raise ValueError('Empty or nonfinite coordinates/weights.')
        points['x']=-points.x.abs()
        points=points.groupby(['x','y','z'],as_index=False,sort=True).weight.mean()
        tree=cKDTree(points[['x','y','z']].to_numpy());weights=points.weight.to_numpy()
        data=np.zeros(np.prod(ref.shape),dtype=np.float32);coverage=np.zeros(data.size,dtype=np.uint8)
        for start in range(0,data.size,100000):
            stop=min(start+100000,data.size)
            ijk=np.column_stack(np.unravel_index(np.arange(start,stop),ref.shape))
            xyz=nib.affines.apply_affine(ref.affine,ijk)
            distance,index=tree.query(xyz)
            valid=(xyz[:,0]<0)&(distance<=radius_mm)
            data[start:stop][valid]=weights[index[valid]]
            coverage[start:stop][valid]=1
        if not coverage.any():raise ValueError(f'{modality}: no voxels covered; check template space and radius.')
        files={}
        for name,values in [('L',data),('R',np.zeros_like(data)),('coverage',coverage)]:
            target=path.parent/f'{modality}_{name}.nii.gz'
            header=ref.header.copy();header.set_data_dtype(values.dtype)
            image=nib.Nifti1Image(values.reshape(ref.shape),ref.affine,header)
            image.header.set_slope_inter(1,0)
            nib.save(image,str(target))
            files[name]=target

        outputs[modality]=files
    return outputs


def map_to_workbench(volumes, left_surface=LEFT_SURFACE, right_surface=RIGHT_SURFACE,
                     wb_command=WB_COMMAND):
    """Map each volume to its hemisphere with the supplied template's trilinear command.

    Open resulting .func.gii metrics together with the corresponding surfaces
    in wb_view. This exports surface data, not rendered screenshots.
    """
    executable=shutil.which(str(wb_command))
    if executable is None:
        raise FileNotFoundError(f'Workbench executable not found: {wb_command}')
    for surface in (left_surface,right_surface):
        if not Path(surface).is_file():raise FileNotFoundError(surface)
    outputs={}
    for modality,files in volumes.items():
        outputs[modality]={}
        for hemisphere,surface in [('L',left_surface),('R',right_surface)]:
            volume=Path(files[hemisphere]);target=volume.with_name(volume.name.removesuffix('.nii.gz')+'.func.gii')
            subprocess.run([executable,'-volume-to-surface-mapping',str(volume),
                            str(surface),str(target),'-trilinear'],check=True)
            outputs[modality][hemisphere]=target
    return outputs


if __name__=='__main__':
    template_path = TEMPLATE_NIFTI
    template_path.parent.mkdir(parents=True, exist_ok=True)

    template = load_mni152_template(resolution=2)  # 2-mm voxels
    template.header.set_xyzt_units("mm")
    template.to_filename(str(template_path))

    for required in (TEMPLATE_NIFTI,LEFT_SURFACE,RIGHT_SURFACE):
        if not required.is_file():
            raise FileNotFoundError(f'Set the template paths at the top of visualization.py: {required}')
    
    if shutil.which(str(WB_COMMAND)) is None:
        raise FileNotFoundError('Set WB_COMMAND to your Workbench executable.')
    
    #weights=compute_weights()
    #volumes=create_nifti(weights)

    volumes = {
        'ieeg' : {'L': OUTPUT_DIR / 'ieeg_L.nii.gz', 
                'R': OUTPUT_DIR / 'ieeg_R.nii.gz', 
                'coverage': OUTPUT_DIR / 'ieeg_coverage.nii.gz'},
        'meg':{'L': OUTPUT_DIR / 'meg_L.nii.gz', 
                'R': OUTPUT_DIR / 'meg_R.nii.gz', 
                'coverage': OUTPUT_DIR / 'meg_coverage.nii.gz'}}
        
    print(map_to_workbench(volumes))
