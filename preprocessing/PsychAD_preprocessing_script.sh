python PsychAD_preprocessing.py \
    --cond_col c02x \
    --donor_col SubID \
    --save_path ../data/PsychAD/c02.h5ad
python PsychAD_preprocessing.py \
    --cond_col c02x \
    --donor_col SubID \
    --pos_cond AD \
    --neg_cond Control \
    --save_path ../data/PsychAD/c02_100v100.h5ad \
    --run_da True \
    --sex_col Sex \
    --subsample_num "100:100"

python PsychAD_preprocessing.py \
    --cond_col r01x \
    --donor_col SubID \
    --save_path ../data/PsychAD/r01.h5ad
python PsychAD_preprocessing.py \
    --cond_col r01x \
    --donor_col SubID \
    --pos_cond 6 \
    --neg_cond 0 \
    --save_path ../data/PsychAD/r01_30v30.h5ad \
    --run_da True \
    --sex_col Sex \
    --subsample_num "30:30"

python PsychAD_preprocessing.py \
    --cond_col c90x \
    --donor_col SubID \
    --save_path ../data/PsychAD/c90.h5ad
python PsychAD_preprocessing.py \
    --cond_col c90x \
    --donor_col SubID \
    --pos_cond Sleep_WeightGain_Guilt_Suicide \
    --neg_cond Control \
    --save_path ../data/PsychAD/c90_63v63.h5ad \
    --run_da True \
    --sex_col Sex \
    --subsample_num "63:63"

python PsychAD_preprocessing.py \
    --cond_col c91x \
    --donor_col SubID \
    --save_path ../data/PsychAD/c91.h5ad
python PsychAD_preprocessing.py \
    --cond_col c91x \
    --donor_col SubID \
    --pos_cond WeightLoss_PMA \
    --neg_cond Control \
    --save_path ../data/PsychAD/c91_30v30.h5ad \
    --run_da True \
    --sex_col Sex \
    --subsample_num "30:30"

python PsychAD_preprocessing.py \
    --cond_col c92x \
    --donor_col SubID \
    --save_path ../data/PsychAD/c92.h5ad
python PsychAD_preprocessing.py \
    --cond_col c92x \
    --donor_col SubID \
    --pos_cond Depression_Mood \
    --neg_cond Control \
    --save_path ../data/PsychAD/c92_100:100.h5ad \
    --run_da True \
    --sex_col Sex \
    --subsample_num "100:100"
