# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
# output=snap/vqa/$name
# mkdir -p $output/src
# cp -r src/* $output/src/
# cp $0 $output/run.bash
output=snap/vqa
# cd ..
# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./fluence \ 
    python fluence/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output
