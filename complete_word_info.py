import pandas as pd 


df_words = pd.read_csv('sentences/lpp_en_regressors_baseline.csv')
# print(df_words.head())

df_words['pref_id'] = None

with open('act_out/lpp_en/prefixes.txt', 'r') as f_pref:
    sections = f_pref.read().strip().split('\n\n')

for i_sec, section in enumerate(sections, start=1):
    words_sec = df_words[df_words['sec_id'] == i_sec]['lemma']  # 'lemma' is the normalized word
    pref_id_sec = []  # the values for the 'pref_id' column of df_words in this section
    
    print(words_sec[:10])
    
    pref_sec = section.strip().split('\n')  # prefixes in this section
    j_pref = 0  # iteration indices for the prefixes
    
    for word in words_sec:
        word = word.lower()
        pref_suffix = pref_sec[j_pref].split('-->')[0].split()[-1]  # the last input token in the prefix
        # print(word, pref_suffix)
        
        while pref_suffix != word:
            # if len(pref_suffix) >= len(word):
            #     raise RuntimeError(f'Mismatching subwords and word: {pref_suffix} vs. {word}')
            j_pref += 1
            pref_suffix = pref_sec[j_pref].split('-->')[0].split()[-1]
            # print(f'suffix: {pref_suffix}')
        
        pref_id_sec.append(j_pref)
        j_pref += 1
        
        # if word == pref_suffix:  # the input token is a whole word and it matches with the human heard word; or the input token is the last subword token of that heard word
        #     pref_id_sec.append(j_pref)
        #     j_pref += 1
            
        # elif word.startswith(pref_suffix):  # the input token is the first subword of the human heard word
        #     pref_id_sec.append(j_pref)
        #     # skip the remaining subwords
        #     while pref_suffix != word:
        #         if len(pref_suffix) >= len(word):
        #             raise RuntimeError(f'Mismatching subwords and word: {pref_suffix} vs. {word}')
        #         j_pref += 1
        #         pref_suffix = pref_sec[j_pref].split('-->')[0].split()[-1]
        #     j_pref += 1
                
        # else:  ## pref_suffix not in word:
        #     raise ValueError(f'Mismatching word and prefixes: {word}, {pref_suffix}')
    
    assert len(pref_id_sec) == len(words_sec)
    df_words.loc[df_words['sec_id'] == i_sec, 'pref_id'] = pref_id_sec


# read the syntactic parsing depths
for parse_method in ['bu', 'lc', 'td']:  # bottom-up, left corner, top-down
    with open(f'regression_targets/lpp_en_{parse_method}.csv', 'r') as f_parse:
        _depths = f_parse.read().strip().split('\n')
        depths = [eval(x) for x in _depths]  # parsing depths
        assert len(depths) == len(df_words)
        
        new_col = f'depth_{parse_method}'
        df_words[new_col] = depths

df_words.to_csv('sentences/lpp_en_word_info.csv', index=False)