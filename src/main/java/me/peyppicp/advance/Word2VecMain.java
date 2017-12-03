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

    public static void main(String[] args) throws IOException {
        BasicLineIterator lineIterator = new BasicLineIterator("F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\distinctLines2.txt");
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(5)
                .layerSize(100)
                .seed(42)
                .windowSize(20)
                .iterate(lineIterator)
                .tokenizerFactory(tokenizerFactory)
                .build();
        vec.fit();

        WordVectorSerializer.writeWordVectors(vec,"word2vec.txt");
        WordVectorSerializer.writeWordVectors(vec.lookupTable(), "word2vecLookUpTable.txt");

//        Collection<String> lst = vec.wordsNearest("day", 10);
//        System.out.println(lst);
    }
}
