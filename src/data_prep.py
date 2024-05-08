import pandas as pd


# Remove outliers using IQR method
def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    return column.between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)


def prepare_data(data_frame: pd.DataFrame) -> pd.DataFrame:

    irrelevant_features = ["Loan_ID"]

    data_frame.drop(
        columns=irrelevant_features,
        inplace=True
    )

    # Handle missing values
    data_frame.fillna({
        'Gender': data_frame['Gender'].mode()[0],
        'Married': data_frame['Married'].mode()[0],
        'Dependents': data_frame['Dependents'].mode()[0],
        'Self_Employed': data_frame['Self_Employed'].mode()[0],
        'LoanAmount': data_frame['LoanAmount'].mean(),
        'Loan_Amount_Term': data_frame['Loan_Amount_Term'].mode()[0],
        'Credit_History': data_frame['Credit_History'].mode()[0]
    }, inplace=True)

    # Removeing outliers
    data_frame = data_frame[remove_outliers(data_frame['ApplicantIncome'])]
    data_frame = data_frame[remove_outliers(data_frame['CoapplicantIncome'])]
    data_frame = data_frame[remove_outliers(data_frame['LoanAmount'])]

    return data_frame


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('./data/raw_data.csv')

    # Prepare data
    preped_df = prepare_data(df)

    # Save prepared data
    preped_df.to_csv("./data/prepared_data.csv", index=False)

    