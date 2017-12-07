package me.peyppicp.advance;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import java.io.IOException;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
public class Word2VecMain {

//    public static final String OUTPUT = "/home/peyppicp/output/";

    public static void main(String[] args) throws IOException {
        String filePath = "new_sample.txt";
        int minWordFrequency = 50;
        int iterations = 5;
        int layerSize = 100;
        int seed = 3543;
        int windowSize = 10;

        BasicLineIterator lineIterator = new BasicLineIterator(filePath);
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(minWordFrequency)
                .iterations(iterations)
                .layerSize(layerSize)
                .seed(seed)
                .windowSize(windowSize)
                .iterate(lineIterator)
                .tokenizerFactory(tokenizerFactory)
                .build();
        vec.fit();

        WordVectorSerializer.writeWordVectors(vec, "word2vec.txt");
        WordVectorSerializer.writeWordVectors(vec.lookupTable(), "word2vecLookUpTable.txt");

//        Collection<String> lst = vec.wordsNearest("day", 10);
//        System.out.println(lst);
    }
}
