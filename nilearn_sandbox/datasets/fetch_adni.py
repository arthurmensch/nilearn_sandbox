# -*- coding: utf-8 -*-
"""
@author: rahim.mehdi@gmail.com
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch


def _set_base_dir():
    """ base_dir
    """
    base_dir = ''
    with open(os.path.join(os.path.dirname(__file__), 'paths.pref'),
              'rU') as f:
        paths = [x.strip() for x in f.read().split('\n')]
        for path in paths:
            if os.path.isdir(path):
                print('Datadir= %s' % path)
                base_dir = path
                break
    if base_dir == '':
        raise OSError('Data not found !')
    return base_dir


def _set_data_base_dir(folder):
    """ base_dir + folder
    """
    return os.path.join(_set_base_dir(), folder)


def _get_subjects_and_description(base_dir,
                                  prefix,
                                  exclusion_file='excluded_subjects.txt',
                                  description_csv='description_file.csv'):
    """  Returns list of subjects and phenotypic dataframe
    """

    # load files and set dirs
    BASE_DIR = _set_data_base_dir(base_dir)
    subject_paths = sorted(glob.glob(os.path.join(BASE_DIR, prefix)))

    fname = os.path.join(BASE_DIR, exclusion_file)
    if not os.path.isfile(fname):
        raise OSError('%s not found ...' % fname)
    excluded_subjects = []
    if os.stat(fname).st_size > 0:
        excluded_subjects = np.loadtxt(fname, dtype=str)

    fname = os.path.join(BASE_DIR, description_csv)
    if not os.path.isfile(fname):
        raise OSError('%s not found ...' % fname)
    description = pd.read_csv(fname)

    # exclude bad QC subjects
    excluded_paths = np.array(map(lambda x: os.path.join(BASE_DIR, x),
                                  excluded_subjects))
    subject_paths = np.setdiff1d(subject_paths, excluded_paths)

    # get subject_id
    subjects = [os.path.split(s)[-1] for s in subject_paths]

    return subjects, subject_paths, description


def _glob_subject_img(subject_path, suffix, first_img=False):
    """ Get subject image (pet, func, ...)
        for a given subject and a suffix
    """

    img_files = sorted(glob.glob(os.path.join(subject_path, suffix)))
    if len(img_files) == 0:
        raise IndexError('Image not found in %s' % subject_path)
    elif first_img:
        return img_files[0]
    else:
        return img_files


def fetch_adni_longitudinal_rs_fmri_DARTEL():
    """ Returns longitudinal func processed with DARTEL
    """
    return fetch_adni_longitudinal_rs_fmri('ADNI_longitudinal_rs_fmri_DARTEL',
                                           'resampled*.nii')


def fetch_adni_longitudinal_rs_fmri(dirname='ADNI_longitudinal_rs_fmri',
                                    prefix='wr*.nii'):
    """ Returns paths of ADNI rs-fMRI
    """

    # get file paths and description
    images, subject_paths, description = _get_subjects_and_description(
                                         base_dir=dirname, prefix='I[0-9]*')
    images = np.array(images)
    # get func files
    func_files = list(map(lambda x: _glob_subject_img(x, suffix='func/' + prefix,
                                                 first_img=True),
                     subject_paths))
    func_files = func_files

    # get phenotype from csv
    df = description[description['Image_ID'].isin(images)]
    df = df.sort('Image_ID')
    dx_group = np.array(df['DX_Group'])
    subjects = np.array(df['Subject_ID'])

    return Bunch(func=func_files, dx_group=dx_group,
                 subjects=subjects, images=images)


def fetch_adni_masks():
    """Returns paths of masks (pet, fmri, both)
    Returns
    -------
    mask : Bunch containing:
           - pet
           - pet_longitudinal
           - fmri
           - fmri_longitudinal
           - petmr
           - petmr_longitudinal
    """
    BASE_DIR = _set_data_base_dir('features/masks')

    return Bunch(pet=os.path.join(BASE_DIR, 'mask_pet.nii.gz'),
                 fmri=os.path.join(BASE_DIR, 'mask_fmri.nii.gz'),
                 pet_longitudinal=os.path.join(BASE_DIR,
                                               'mask_longitudinal_fdg_pet'
                                               '.nii.gz'),
                 petmr=os.path.join(BASE_DIR, 'mask_petmr.nii.gz'),
                 petmr_longitudinal=os.path.join(BASE_DIR,
                                                 'mask_longitudinal_petmr'
                                                 '.nii.gz'),
                 fmri_longitudinal=os.path.join(BASE_DIR,
                                                'mask_longitudinal_fmri'
                                                '.nii.gz'))
