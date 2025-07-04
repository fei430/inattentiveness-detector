@echo off
call .\venv310\Scripts\activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python spiga/demo/app.py --input 0 --tracker RetinaSort --show fps face_id landmarks headpose
pause