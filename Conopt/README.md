# Run the code
To run the code, we have to input the following code:

For consensus optimization
```console
$ python run.py --image-size=375 --output=32 --c-dim=3 --z-dim=256 --gf-dim=64 --df-dim=64 --reg-param=10. --g-architecture=dcgan3_nobn --d-architecture=dcgan3_nobn --gan-type=standard --optimizer=conopt 
```

For Simultaneous Gradient Ascent/Descent
```console
$ python run.py --image-size=375 --output=32 --c-dim=3 --z-dim=256 --gf-dim=64 --df-dim=64 --reg-param=10. --g-architecture=dcgan3 --d-architecture=dcgan3 --gan-type=standard --optimizer=simga 
```

The result is shown as ppt https://github.com/limcherhang/finalreport/blob/main/structural_ML_final_report.pdf
