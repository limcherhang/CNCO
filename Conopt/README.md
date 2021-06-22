# Run the code
To run the code, we have to input the following code:

For consensus optimization
```console
$ python run.py --image-size=375 --output=32 --c-dim=3 --z-dim=256 --gf-dim=64 --df-dim=64 --reg-param=10. --g-architecture=dcgan4_nobn --d-architecture=dcgan4_nobn --gan-type=standard --optimizer=conopt 
```

For Simultaneous Gradient Ascent/Descent
```console
$ python run.py --image-size=375 --output=32 --c-dim=3 --z-dim=256 --gf-dim=64 --df-dim=64 --reg-param=10. --g-architecture=dcgan4 --d-architecture=dcgan4 --gan-type=standard --optimizer=simga 
```
