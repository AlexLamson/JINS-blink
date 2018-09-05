
rem stationary EOG+IMU
echo 0 1 1 > data_settings.txt
python train_machine_learning_model.py

rem stationary IMU
echo 0 0 1 > data_settings.txt
python train_machine_learning_model.py

rem stationary EOG
echo 0 1 0 > data_settings.txt
python train_machine_learning_model.py

rem mobile EOG+IMU
echo 1 1 1 > data_settings.txt
python train_machine_learning_model.py

rem mobile IMU
echo 1 0 1 > data_settings.txt
python train_machine_learning_model.py

rem mobile EOG
echo 1 1 0 > data_settings.txt
python train_machine_learning_model.py

