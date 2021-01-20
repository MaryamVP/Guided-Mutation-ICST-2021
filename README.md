# Guided-Mutation-ICST-2021

Over the past few years, deep neural networks (DNNs) have been continuously expanding their real-world applications for source code processing tasks across the software engineering domain, e.g., clone detection, code search, comment generation. Although quite a few recent works have been performed on testing of DNNs in the context of image and speech processing, limited progress has been achieved so far on DNN testing in the context of source code processing, that exhibits rather unique characteristics and challenges.

In this paper, we propose a search-based testing framework for DNNs of source code embedding and its downstream processing tasks like Code Search. To generate new test inputs, we adopt popular source code refactoring tools to generate the semantically equivalent variants. For more effective testing, we leverage the DNN mutation testing to guide the testing direction. To demonstrate the usefulness of our technique, we perform a large-scale evaluation on popular DNNs of source code processing based on multiple state-of-the-art code embedding methods (i.e., Code2vec, Code2seq  and CodeBERT). The testing results show that our generated adversarial samples can on average reduce the performance of these DNNs from 5.41% to 9.58%. Through retraining the DNNs with our generated adversarial samples, the robustness of DNN can improve by 23.05% on average. The evaluation results also show that our adversarial test generation strategy has the least negative impact (median of 3.56%), on the performance of the DNNs for regular test data, compared to the other methods.

All required infromation regarding the original models can be found in their Github page, as follow:

    - Code2vec:https://github.com/tech-srl/code2vec
    - Code2seq:https://github.com/tech-srl/code2seq
    -CodeBERT:https://github.com/microsoft/CodeBERT
