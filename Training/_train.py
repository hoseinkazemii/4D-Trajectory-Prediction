def _train(X_train, y_train, model, **params):
    verbose = params.get("verbose")
    num_epochs = params.get("num_epochs")
    batch_size = params.get("batch_size")
    validation_split = params.get("validation_split")
    warmup = params.get("warmup")

    if verbose:
        print("training the model...")

    if not warmup:
        history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=validation_split)
        model.save('./SavedModels/seq2seq_trajectory_model_3d.h5')
        return history, model