import pandas as pd
import numpy as np

class Data_processor:
    @staticmethod
    def clip_outliers(df, column, min_val, max_val):
        """
        Обрезает выбросы в указанном столбце до заданных границ
        
        Args:
            df: исходный DataFrame
            column: название столбца для обработки
            min_val: минимальное значение (нижняя граница)
            max_val: максимальное значение (верхняя граница)
            
        Returns:
            DataFrame с обработанными значениями
        """
        df[column] = df[column].clip(lower=min_val, upper=max_val)
        return df
