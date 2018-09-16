python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=3 --exp_name='Base line with SGD(0.001)'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=3 --exp_name='Base line with Adam(0.001)'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=3 --exp_name='Base line with RMS(0.001)'

python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=3 --include_bn=true --exp_name='SGD(0.001) + BN'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=3 --include_bn=true --exp_name='Adam(0.001) + BN'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=3 --include_bn=true --exp_name='RMS(0.001) + BN'

python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=3 --activation=elu --exp_name='SGD(0.001) + ELU'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=3 --activation=elu --exp_name='Adam(0.001) + ELU'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=3 --activation=elu --exp_name='RMS(0.001) + ELU'

python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=3 --activation=elu --include_bn=true --exp_name='SGD(0.001) + ELU + BN'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=3 --activation=elu --include_bn=true --exp_name='Adam(0.001) + ELU + BN'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=3 --activation=elu --include_bn=true --exp_name='RMS(0.001) + ELU + BN'

python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=2 --exp_name='Base line with SGD(0.001) Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=2 --exp_name='Base line with Adam(0.001) Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=2 --exp_name='Base line with RMS(0.001) Pool=2'

python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=2 --include_bn=true --exp_name='SGD(0.001) + BN + Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=2 --include_bn=true --exp_name='Adam(0.001) + BN + Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=2 --include_bn=true --exp_name='RMS(0.001) + BN + Pool=2'

python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=2 --activation=elu --exp_name='SGD(0.001) + ELU + Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=2 --activation=elu --exp_name='Adam(0.001) + ELU + Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=2 --activation=elu --exp_name='RMS(0.001) + ELU + Pool=2'

python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=sgd --pool_size=2 --activation=elu --include_bn=true --exp_name='SGD(0.001) + ELU + BN + Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=adam --pool_size=2 --activation=elu --include_bn=true --exp_name='Adam(0.001) + ELU + BN + Pool=2'
python estimator.py --n_epochs=1 --batch_size=128 --lr=0.001 --optimizer=rms --pool_size=2 --activation=elu --include_bn=true --exp_name='RMS(0.001) + ELU + BN + Pool=2'
