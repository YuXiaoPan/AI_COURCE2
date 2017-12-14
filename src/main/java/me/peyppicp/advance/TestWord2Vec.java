package me.peyppicp.advance;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.util.Collection;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
public class TestWord2Vec {

    public static void main(String[] args) {
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("glove.twitter.27B.100d.txt");
        Collection<String> strings = word2Vec.wordsNearest("i'm", 10);
        Collection<String> strings1 = word2Vec.wordsNearest("im", 10);
        Collection<String> strings2 = word2Vec.wordsNearest("'m", 10);
        System.out.println(strings);
        System.out.println(strings1);
        System.out.println(strings2);
//        int[] ints = new int[100];
//        Arrays.stream(ints).parallel().forEach(System.out::println);
    }
}
