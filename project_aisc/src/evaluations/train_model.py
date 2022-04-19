import sys
sys.path.append('src/data')
sys.path.append('src/features')
sys.path.append('src/model')
import make_dataset
import build_features
import build_models
import tensorflow as tf


def main():
    #Load atomic data
    ptable = make_dataset.PeriodicTable()
    #Initialize the processor for atomic data
    atom_processor = build_features.AtomData(ptable)
    #Process atomic data
    atom_processed = atom_processor.get_atom_data()

    #Load SuperCon dataset
    sc_dataframe = make_dataset.SuperCon(sc_path ='data/raw/supercon_tot.csv')
    #Initialize processor for SuperCon data
    supercon_processor = build_features.SuperConData(atom_processed,sc_dataframe,padding = 10)
    #Process SuperCon data
    supercon_processed = supercon_processor.get_dataset()

    tc_regression = sc_dataframe['critical_temp']
    X,X_test,Y,Y_test = build_features.train_test_split(supercon_processed,tc_regression,0.2)
    X,X_val,Y,Y_val = build_features.train_test_split(X,Y,0.2)

    #Define model and train it
    model = build_models.get_model(model='regressor')
    callbacks = [tf.keras.callbacks.EarlyStopping(min_delta=5,patience = 40,restore_best_weights=True)]
    model.fit(X,Y,validation_data=(X_val,Y_val),epochs=200,callbacks=callbacks)

    #Save scores and metrics' name
    score = model.evaluate(X_test,Y_test,verbose=0)
    metrics_name = [metric.name for metric in model.metrics]
    #Print the metric and the relative score of the model
    for name,value in zip(metrics_name,score):
        print(name+':',value)

if __name__ == '__main__':
    main()
