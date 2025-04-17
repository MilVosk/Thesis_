from few_shot import *
from Thesis_.file_reader import train

sampled, test = balanced_sample(train, train['Label'], 10)
prompt = prompt_generator(sampled)
result_df = generate_gpt_response_with_relations(prompt, test[41:60])

result_df['Gold'] = test[41:60]['Label'].reset_index(drop=True)
result_df

true = [str(i) for i in result_df['Gold']]
predicted = [str(j) for j in result_df['Predicted_Label']]


result_df.to_excel("output_with_f1_0.618.xlsx", index=False)