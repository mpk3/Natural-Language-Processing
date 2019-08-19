### DocInspect<h1>
Simple script I wrote to look at some features of corpora including token and POS counts. I mostly use it in ipython. I will continue to expand on its features and _prettify_ the graph output.
#### Exclaimer:<h2>
The create_data() function can be a little slow. This should really only be used for single documents until I create a more robust version maybe leveraging matrix operations. Also, this current version does not offer any current token filters, aka stopwords have not been removed. I will look at adding function to fix this.

#### Example:<h2>
- Comes from inspecting Jack London's 'Call of the Wild' and a translated copy of Goethe's 'Faust', both obtained through gutenberg

```python
ipython: run doc_inspect.py
>df, tok, pos = standard_make(['test_data/CofW.txt'])
>tokplot = plot_nlargest(tok['count'])
>plt.show()

![Top 50 Tokens in Call of the Wild](Top50CofW.png)



ipython: run doc_inspect.py
> df = create_data(path-to-filein) 
> df_tok_c = inv_index_token(df)
> df_pos_c = inv_index_POS(df)

>tokplot = plot_nlargest(df_tok_c['count'], 5, 0)
>posplot = plot_nlargest(df_pos_c['count'], 5, 0)
>plt.show()
```
