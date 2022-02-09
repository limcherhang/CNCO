I'm a newbie researcher in Taiwan, if I have any wrong idea, please contact me. Thank you

This is my prototype report and contains implemented on two paper:
  1. https://arxiv.org/abs/1805.05751
  2. https://arxiv.org/abs/1705.10461

# New idea

By the two methods proposed in the paper above, I have a new idea that combine two methods and do some simple simulation. By the way, I updated implement on the datasets mnist.

# Run the code
To run the code, 

``` console
$ python main.py --opt='SimGA' --epochs=100 --lr=1e-4 --batch_size=32 --is_fid=1 --fid_range=50
$ python main.py --opt='SimGA' --epochs=100 --lr=1e-3 --batch_size=5000 --is_fid=1 --fid_range=50
$ python main.py --opt='ConOpt' --epochs=100 --lr=1e-4 --batch_size=32 --is_fid=1 --fid_range=50
$ python main.py --opt='ConOpt' --epochs=100 --lr=1e-3 --batch_size=5000 --is_fid=1 --fid_range=50
$ python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50 
$ python main.py --opt='CESP' --epochs=100 --lr=1e-3 --batch_size=5000 --alpha=5.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.1 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.1 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.1 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.5 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.5 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=0.5 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=1.0 --alpha=0.1 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=1.0 --alpha=0.5 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
$ python main.py --opt='CNCO' --epochs=100 --lr=1e-3 --batch_size=5000 --gamma=1.0 --alpha=1.0 --is_eigen=1 --is_fid=1 --load_model_path='' --fid_range=50
```

And get the result of loss function and generated image on Simultaneous Gradient Descent(SimGA), Consensus Optimization(ConOpt), Negative Curvature Exploitation for Local Saddle Point Problem(CESP) and Consensus Negative Curvature Optimization(CNCO).

More result details will updated in future in my paper.
