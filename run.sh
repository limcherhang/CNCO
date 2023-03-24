python main.py --opt='SimGA' --epochs=100 --lr=1e-4 --batch_size=32 --is_fid=1 --fid_range=50
python main.py --opt='SimGA' --epochs=100 --lr=1e-3 --batch_size=5000 --is_fid=1 --fid_range=50
python main.py --opt='ConOpt' --epochs=100 --lr=1e-4 --gamma=0.1 --batch_size=32 --is_fid=1 --fid_range=50
python main.py --opt='ConOpt' --epochs=100 --lr=1e-4 --gamma=0.5 --batch_size=32 --is_fid=1 --fid_range=50
python main.py --opt='ConOpt' --epochs=100 --lr=1e-4 --gamma=1.0 --batch_size=32 --is_fid=1 --fid_range=50
python main.py --opt='ConOpt' --epochs=100 --lr=1e-3 --gamma=0.1 --batch_size=5000 --is_fid=1 --fid_range=50
python main.py --opt='ConOpt' --epochs=100 --lr=1e-3 --gamma=0.5 --batch_size=5000 --is_fid=1 --fid_range=50
python main.py --opt='ConOpt' --epochs=100 --lr=1e-3 --gamma=1.0 --batch_size=5000 --is_fid=1 --fid_range=50
python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50 
python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=5.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.1 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.1 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.1 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.5 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.5 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.5 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=1.0 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=1.0 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=1.0 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50