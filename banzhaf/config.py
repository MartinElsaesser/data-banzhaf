big_dataset = [
    "MNIST",
    "CIFAR10",
    # 'Dog_vs_Cat',        # not implemented
    # 'Dog_vs_CatFeature', # not implemented
    "FMNIST",
]

OpenML_dataset = ["fraud", "apsfail", "click", "phoneme", "wind", "pol", "creditcard", "cpu", "vehicle", "2dplanes"]

allowed_value_types = [
    "Uniform",
    "Shapley_Perm",
    "Banzhaf_GT",
    "BetaShapley",
    "LOO",
    "FixedCard_MC",
    "FixedCard_MSR",
    "FixedCard_MSRPerm",
    "KNN",
    "Shapley_GT",
    "LeastCore",
]
