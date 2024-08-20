# Training:
#    you can find the trained models in the "save_models" folder
python train.py --train_dir ./data_split/One_train.txt --test_dir ./data_split/One_test.txt

# Testing
#    you can find the results in the "vis_results" and "output_results" folders
#    update the path of the pre-trained model to test a different model
python test.py --ckpt ./save_models/08-21-00-56/last.pt --test_dir ./data_split/One_test.txt
