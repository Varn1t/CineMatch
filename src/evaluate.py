from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import pandas as pd


def evaluate_rmse(ratings_df):
    """
    5-fold cross-validation RMSE for SVD.
    """
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]], reader
    )
    model = SVD(n_factors=100, n_epochs=30, lr_all=0.007, reg_all=0.05, random_state=42)
    results = cross_validate(model, data, measures=["RMSE"], cv=5, verbose=False)
    mean_rmse = results["test_rmse"].mean()
    print(f"Mean RMSE (5-fold CV): {mean_rmse:.4f}")
    return mean_rmse


def evaluate_precision_recall_at_k(ratings_df, k=10, threshold=4.0):
    """
    Computes Precision@K and Recall@K for SVD across all users.
    """
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]], reader
    )

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=100, n_epochs=30, lr_all=0.007, reg_all=0.05, random_state=42)
    model.fit(trainset)
    predictions = model.test(testset)

    # Group predictions by user
    user_predictions = {}
    for pred in predictions:
        user_predictions.setdefault(pred.uid, []).append(pred)

    precisions, recalls = [], []

    for uid, preds in user_predictions.items():
        # Sort by estimated rating descending, take top K
        preds.sort(key=lambda x: x.est, reverse=True)
        top_k = preds[:k]

        # Relevant = actually rated >= threshold in test set
        n_relevant = sum(1 for p in preds if p.r_ui >= threshold)
        n_recommended_relevant = sum(1 for p in top_k if p.r_ui >= threshold)

        precision = n_recommended_relevant / len(top_k)  # users can have < k test items
        recall = n_recommended_relevant / n_relevant if n_relevant > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)

    print(f"Precision@{k}: {mean_precision:.4f}")
    print(f"Recall@{k}:    {mean_recall:.4f}")

    return mean_precision, mean_recall