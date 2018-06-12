CUDA_VISIBLE_DEVICES=0 python nmt.py \
    --cuda \
    --mode test \
    --load_model /path/to/trained/model \
    --save_to_file output \
    --test_src /path/to/valid/article \
    --test_tgt /path/to/valid/title
