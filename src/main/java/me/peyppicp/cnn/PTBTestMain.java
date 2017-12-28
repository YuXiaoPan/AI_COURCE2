package me.peyppicp.cnn;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;

/**
 * @author YuXiao Pan
 * @date 2017/12/28
 * @email yuxiao.pan@kikatech.com
 */
public class PTBTestMain {

    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("./src/main/resources/model/ptb0.txt");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel("glove.twitter.27B.50d.txt");
        WordToIndex wordToIndex = new WordToIndex("pair.txt");
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        RnnTest rnnTest = new RnnTest(wordVectors, wordToIndex, tokenizerFactory, model, null);
        rnnTest.generateTokensFromStr("happy birthday to you");
    }
}
