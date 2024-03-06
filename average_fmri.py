import numpy as np

for sec_id in range(1, 10):
    # read the fmri data of all subjects in this section
    fmri_sec = []  # n_subj
    
    for subj_id in range(1, 50):
        subj = f'sub-EN00{subj_id}' if subj_id < 10 else f'sub-EN0{subj_id}'
        fmri_sec_subj = np.load(f'lpp_surface/subjects/{subj}/{subj}-run-{sec_id}-fsaverage7-lh.nii.gz.npy')  #  (n_vertex, n_timepoint_in_section)
        fmri_sec.append(fmri_sec_subj)
    
    avg_fmri_sec = np.mean(np.stack(fmri_sec, axis=0), axis=0)  # (n_vertex, n_timepoint_in_section)
    
    print(avg_fmri_sec.shape)
    np.save(f'lpp_surface/average/avg-run-{sec_id}-fsaverage7-lh.nii.gz.npy', avg_fmri_sec)