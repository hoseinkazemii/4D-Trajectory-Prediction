def _df_to_array(df, **params):
    verbose = params.get("verbose")
    if verbose:
        print("converting dataframe to array...")
    data = df[['X', 'Y', 'Z']].values
    return data