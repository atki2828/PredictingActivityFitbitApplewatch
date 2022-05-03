import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from itertools import product
from typing import Tuple, List, Union, Dict

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Class to wrap grid_search and create pipeline steps
class ClassifierPipeLine:
    """Class to run gridsearch pipeline for
    classifiers and display confusion matrix and
    classification report.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        clf,
    ):
        """Initialize Object

        Args:
            X_train (_type_): training predictors
            y_train (_type_): training response
            X_test (_type_): test predictors
            y_test (_type_): test response
            clf (_type_): classifer
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf = clf

        self.pipeline = None
        self.grid_cv_fit = None

    def create_pipeline(
        self, steps: Union[List[Tuple[str, str]], None] = None, output: bool = False
    ) -> Union[None, Pipeline]:
        """Takes in steps for pipeline default is set to None
        If no steps are given will use classifier as last step
        automatically.

        Args:
            steps (Union[List[Tuple[str,str]],None], optional): _description_. Defaults to None.
            output (bool, optional): _description_. Defaults to False.

        Returns:
            Union[None,Pipeline]: _description_
        """
        if steps:
            steps.append(("clf", self.clf))
        else:
            steps = [("clf", self.clf)]
        self.pipeline = Pipeline(steps)
        return self.pipeline if output else None

    def create_grid_search(
        self, param_grid: Dict[str, str], cv: int, scoring: str, output: bool = False
    ) -> Union[GridSearchCV, None]:
        """Create gridsearch with cross validation

        Args:
            param_grid (Dict[str, str]): Parameters for the grid search with the format clf__<param>
            cv (int): number of folds for k fold cv
            scoring (str): scoring measure of fiit
            output (bool, optional): if function should return grid search Defaults to False.

        Returns:
            Union[GridSearchCV, None]: _description_
        """
        model_obj = (
            self.pipeline if self.pipeline else self.create_pipeline(output=True)
        )
        grid_cv = GridSearchCV(model_obj, param_grid=param_grid, cv=cv, scoring=scoring)
        self.grid_cv_fit = grid_cv.fit(self.X_train, self.y_train)
        return self.grid_cv_fit if output else None

    def display_confusion_matrix(
        self, font_scale: float, fig_size: Tuple[int, int]
    ) -> None:
        """Takes in font_scale and fig_size and displays confusion matrix
        of y_test and clf.predict(X_test)

        Args:
            font_scale (float): the font scale
            fig_size (Tuple[int,int]): figsize
        """

        sns.set(font_scale=font_scale)
        fig, ax = plt.subplots(figsize=fig_size)
        cmp = ConfusionMatrixDisplay(
            confusion_matrix(self.y_test, self.grid_cv_fit.predict(self.X_test)),
            display_labels=self.y_test.sort_values().unique(),
        )
        cmp.plot(ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Confusion Matrix for {type(self.clf).__name__}")
        plt.tight_layout()

    def display_classification_report(self) -> None:
        """Prints Classification Report"""
        print(classification_report(self.y_test, self.grid_cv_fit.predict(self.X_test)))


# strip_box plot
def grid_strip_box(
    rows: int, cols: int, figsize: Tuple[int, int], var_list: List[str], **params
) -> None:
    """Creates a Strip plot inside a box plot
    Used to analyze multiple predictors across target
    levels of activity in facetgrid

    Args:
        rows (int): number of rows
        cols (int): number of columns
        figsize (Tuple[int,int]): figsixe
        var_list (List[str]): list of predictors
    """
    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    # if statement to take care of edge case where ax will not be a tuple
    if (rows > 1) & (cols > 1):
        l1 = list(range(rows))
        l2 = list(range(cols))

        axes_list = list(product(l1, l2))
    else:
        axes_list = list(range(max([rows, cols])))

    for y_var, ax_tup in zip(var_list, axes_list):

        sns.stripplot(
            y=y_var,
            size=5,
            jitter=0.35,
            palette="tab10",
            edgecolor="black",
            linewidth=1,
            ax=ax[ax_tup],
            **params,
        )
        sns.boxplot(y=y_var, palette=["#D3D3D3"], linewidth=5, ax=ax[ax_tup], **params)
        ax[ax_tup].set_title(f"Distribution of {y_var} by Activity")
        ax[ax_tup].tick_params(axis="x", labelrotation=45)
        plt.tight_layout()
    plt.show()

    # Standard Scaler function


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function to scale a dataframe

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: scaled dataframe
    """
    scaler = StandardScaler()
    scaler.fit(df)
    return pd.DataFrame(scaler.transform(df))


# Rerturn inertia for K Means Elbow Plot


def calculate_inertia(k: int, df: pd.DataFrame) -> float:
    """Calculate the intertia from a k means
    used to iterate over array of k values
    for elbow plot

    Args:
        k (int): K in KMeans
        df (pd.DataFrame): Data

    Returns:
        float: _description_
    """
    scaled_df = scale_data(df)
    model = KMeans(n_clusters=k)
    model.fit(scaled_df)
    return model.inertia_
