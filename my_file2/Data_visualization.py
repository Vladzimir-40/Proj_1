import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Data_visualization:
    @staticmethod
    def pairplot(df, target_column, numerical_columns=None):
        """Парные сравнения числовых показателей с выделением целевой переменной"""
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        sns.pairplot(df, vars=numerical_columns, hue=target_column)
        plt.show()
    
    @staticmethod
    def heatmap(df, target_column, numerical_columns=None):
        """Тепловая карта корреляций числовых показателей"""
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Добавляем целевую переменную, если она числовая
        if target_column not in numerical_columns and df[target_column].dtype in ['int64', 'float64']:
            numerical_columns.append(target_column)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.show()
    
    @staticmethod
    def boxplot(df, target_column, numerical_columns=None):
        """Ящики с усами для числовых показателей по категориям целевой переменной"""
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Создаем сетку графиков
        n_cols = 3
        n_rows = (len(numerical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_columns):
            sns.boxplot(data=df, x=target_column, y=col, ax=axes[i])
            axes[i].set_title(f'{col} by {target_column}')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Скрываем пустые графики
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()
