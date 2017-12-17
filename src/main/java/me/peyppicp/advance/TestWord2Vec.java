package me.peyppicp.advance;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
public class TestWord2Vec {

    public static void main(String[] args) {
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("glove.twitter.27B.100d.txt");
        WeightLookupTable weightLookupTable = word2Vec.lookupTable();
        System.out.println(word2Vec.getMinWordFrequency());
//        Collection<String> strings2 = word2Vec.wordsNearest(ImmutableList.of("i'm", "happy"), ImmutableList.of(), 10);
//        System.out.println(strings2);
//        int[] ints = new int[100];
//        Arrays.stream(ints).parallel().forEach(System.out::println);
    }
}
