import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as cnf
from itertools import combinations
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelBinarizer

def check_df(dataframe, head=5,non_numeric=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### DESCRIBE #####################")
    print(dataframe.describe())
    
    for col in dataframe.columns:
        if dataframe[col].isna().sum() <= 0:
            if dataframe[col].nunique() > 20:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()}\n')
            else:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- unique: {dataframe[col].unique()}\n')
        else:
            if dataframe[col].nunique() > 20:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- nan: {dataframe[col].isna().sum()}\n')
            else:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- unique: {dataframe[col].unique()} --- nan: {dataframe[col].isna().sum()}\n')
    
    if non_numeric:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
    
def grab_col_names(dataframe, cat_th=10, car_th=20,p=False):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    if p:
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols,cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def do_Target_spareted(dataframe):
    new_spareted_cabin = dataframe[cnf.target].str.split(pat = ",", expand = True)
    
    dataframe.drop(cnf.target, axis=1, inplace=True)
    
    new_spareted_cabin.rename(columns={0 : 'first_menu',
                                       1 : 'second_menu',
                                       2 : 'third_menu'}, inplace=True)
    
    return pd.concat([dataframe, new_spareted_cabin], axis=1)

def sample_sub(ypred):
    
    sample = pd.read_csv("csv_sample.csv")

    submission = pd.DataFrame({"id": sample["id"],
                                cnf.target : ypred})                        
    submission.to_csv("34.csv", index=False)

def binarize_column(column):
    lb = LabelBinarizer()
    transformed_data = lb.fit_transform(column)
    if column.name == "second_menu":
        transformed_data = [np.insert(row, 4, 0) for row in transformed_data]
    elif column.name == "third_menu":
        transformed_data = [np.insert(row, 2, 0) for row in transformed_data]
    return pd.Series([row.tolist() for row in transformed_data])

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False,q1 = 0.01,q3 = 0.99):
    low, up = outlier_thresholds(dataframe, col_name,q1=q1,q3 = q3)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
    
def num_summary(dataframe, numerical_col, plot=False):
    """
        Numerik kolonlar input olarak verilmelidir.
        Sadece ekrana cikti veren herhangi bir degeri return etmeyen bir fonksiyondur.
        For dongusuyle calistiginda grafiklerde bozulma olmamaktadir.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
      
def plot_distributions(dataframe, columns,kde=False, log_transform=False, label_angle=0, 
                       figsize = (8,3) , order_cats= False, target_pie=False, alert=False): 

    if alert == True:
        pie_palette = cnf.alert_palette
    else:
        pie_palette = cnf.sequential_palette
        
    if target_pie == True:
        ax = dataframe[columns].value_counts().plot.pie(autopct='%1.1f%%',
                                              textprops={'fontsize':10},
                                              colors=pie_palette
                                              ).set_title(f"{cnf.target} Distribution")
        plt.ylabel('')
        plt.show()

    else:
        for col in columns:
            if log_transform == True:
                x = np.log10(dataframe[col])
                title = f'{col} - Log Transformed'
            else:
                x = dataframe[col]
                title = f'{col}'
            
            if order_cats == True:
                
                print(pd.DataFrame({col: dataframe[col].value_counts(),
                            "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)}))
            
                print("##########################################")
                
                print(f"NA in {col} : {dataframe[col].isnull().sum()}")
                
                print("##########################################")

                labels = dataframe[col].value_counts(ascending=False).index
                values = dataframe[col].value_counts(ascending=False).values
                
                plt.subplots(figsize=figsize)
                plt.tight_layout()
                plt.xticks(rotation=label_angle)
                sns.barplot(x=labels,
                            y=values,
                            palette = cnf.sequential_palette)
                        
            else:   
            
                quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
                print(dataframe[col].describe(quantiles).T)

                plt.subplots(figsize=figsize)
                plt.tight_layout()
                plt.xticks(rotation=label_angle)
                sns.histplot(x,
                        bins=50,
                        kde=kde,
                        color = cnf.sequential_palette[0])

    
            plt.title(title)
            plt.show()

def numcols_target_corr(dataframe, num_cols,target = cnf.target):
    numvar_combinations = list(combinations(num_cols, 2))
    
    for item in numvar_combinations:
        
        plt.subplots(figsize=(8,4))
        sns.scatterplot(x=dataframe[item[0]], 
                        y=dataframe[item[1]],
                        hue=dataframe[target],
                        palette=cnf.bright_palette
                       ).set_title(f'{item[0]}   &   {item[1]}')
        plt.grid(True)
        plt.show()            
        
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def cat_summary(dataframe, col_name, plot=False,info=True):
    if info==True:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
