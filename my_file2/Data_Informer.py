import pandas as pd

class Data_Informer:
    def __init__(self, df):
        self.df = df
    
    def show_head(self, n=5):
        """Показать первые n строк датафрейма"""
        print("Dataset head:")
        return self.df.head(n)
    
    def show_info(self):
        """Показать информацию о датафрейме"""
        print("Dataset info:")
        return self.df.info()
    
    def show_describe(self):
        """Показать статистическое описание данных"""
        print("Dataset describe:")
        return self.df.describe()
    
    def show_dtypes(self):
        """Показать типы данных столбцов"""
        print("Dataset types:")
        return self.df.dtypes
    
    def show_null_counts(self):
        """Показать количество null значений"""
        print("Dataset null values:")
        return self.df.isnull().sum()
    
    def show_nan_counts(self):
        """Показать количество NaN значений"""
        print("Dataset NaN values:")
        return self.df.isna().sum()
