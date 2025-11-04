CUDA_VISIBLE_DEVICES=0 python run.py pad_ufes_20 >> logs/pad_ufes_20.log
CUDA_VISIBLE_DEVICES=0 python run.py cbis_ddsm mass >> logs/cbis_ddsm_mass.log
CUDA_VISIBLE_DEVICES=0 python run.py cbis_ddsm calc >> logs/cbis_ddsm_calc.log
CUDA_VISIBLE_DEVICES=0 python run.py petfinder-adoption-prediction image >> logs/petfinder-image.log
CUDA_VISIBLE_DEVICES=0 python run.py petfinder-adoption-prediction all >> logs/petfinder-all.log