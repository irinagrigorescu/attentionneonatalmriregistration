# Attention-driven multi-channel deformable registration of structural and microstructural neonatal data

#### Author: Irina Grigorescu   |   irina[_dot_]grigorescu[_at_]kcl[_dot_]ac[_dot_]uk

This is the companion code for the following publications:

- Grigorescu, I. et al. (2021). _Uncertainty-Aware Deep Learning Based Deformable Registration_. UNSURE 2021. LNCS (Springer)
	- [doi.org/10.1007/978-3-030-87735-4_6](https://doi.org/10.1007/978-3-030-87735-4_6)

- Grigorescu, I. et al. (2022). _Attention-Driven Multi-channel Deformable Registration of Structural and Microstructural Neonatal Data_. PIPPI 2022. LNCS (Springer)
	- [doi.org/10.1007/978-3-031-17117-8_7](https://doi.org/10.1007/978-3-031-17117-8_7)


### Example train/validation/test file (one subject / line, see ```example_csv_files``` folder)

```
t2w,lab,dti,fa,ga,as,gender
subj1_T2_img.nii.gz,subj1_tissue_labels.nii.gz,subj1_DTI.nii.gz,subj1_FA.nii.gz,41.0,41.14285714,Male
subj2_T2_img.nii.gz,subj2_tissue_labels.nii.gz,subj2_DTI.nii.gz,subj2_FA.nii.gz,40.14285714,40.28571429,Female
```

Also, bare in mind that the code currently expects the suffix ```_T2_img.nii.gz``` for the images, 
the suffix ```_tissue_labels.nii.gz``` for the tissue maps, 
the suffix ```_DTI.nii.gz``` for the diffusion tensor images,
or ```_FA.nii.gz``` for the fractional anisotropy maps.
It also expects ```_ic_m.nii.gz``` for the internal capsule label.