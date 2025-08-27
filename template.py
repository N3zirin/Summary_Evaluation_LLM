Consistency_prompt_CoT= \
"""Decide if the following summary is consistent with the corresponding article.
      Note that consistency means all information in the summary is supported by the article.
      Explain your reasoning step by step first, and then answer (consistent or inconsistent) at the end:
      <Article>
      {article}
      </Article>

      <Summary>
      {summary}
      </Summary>

      Answer:"""

Consistency_prompt= \
"""Decide if the following summary is consistent with the corresponding article.
      Note that consistency means all information in the summary is supported by the article.
      Do not give any reasoning just answer (consistent or inconsistent) at the end:
      <Article>
      {article}
      </Article>

      <Summary>
      {summary}
      </Summary>

      Answer:"""

Ranking_Prompt_CoT = \
"""Decide which one of the following summary is consistent with the corresponding article.
        Note that consistency means all information in the summary is supported by the article.
        Explain your reasoning step-by-step and then give the answer in <Answer>(A or B)</Answer> tags:

        <Article>
        {document}
        </Article>

        <Summary A>
        {sum_a}
        </Summary A>
        <Summary B>
        {sum_b}
        </Summary B>
        """

Ranking_Prompt = \
"""Decide which one of the following summary is consistent with the corresponding article.
        Note that consistency means all information in the summary is supported by the article.
        Do not explain reasoning steps. Just provide the answer in <Answer>(A or B)</Answer> tags:

        <Article>
        {document}
        </Article>

        <Summary A>
        {sum_a}
        </Summary A>
        <Summary B>
        {sum_b}
        </Summary B>
        """

Summary_Scorring_Prompt_CoT = \
"""Your task is to rate the factual consistency of the provided summary against the source article.A score of 1.0 means the summary is perfectly factual, containing no information that contradicts or is not supported by the article.
        A score of 0.0 means the summary is completely non-factual.
        Carefully read the article and the summary. Then, provide ONLY the numerical factuality score as your response. Add explanation, commentary, or conversational text.
        [ARTICLE]:
        {article}
        [SUMMARY]:
        {summary}
        [FACTUALITY SCORE]:
        """

Summary_Scorring_Prompt = \
"""Your task is to rate the factual consistency of the provided summary against the source article.A score of 1.0 means the summary is perfectly factual, containing no information that contradicts or is not supported by the article.
        A score of 0.0 means the summary is completely non-factual.
        Carefully read the article and the summary. Then, provide ONLY the numerical factuality score as your response. Do not add any explanation, commentary, or conversational text.
        [ARTICLE]:
        {article}
        [SUMMARY]:
        {summary}
        [FACTUALITY SCORE]:
        """