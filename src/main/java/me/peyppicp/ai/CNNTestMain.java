package me.peyppicp.ai;

import com.google.common.base.Preconditions;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author YuXiao Pan
 * @date 2017/12/29
 * @email yuxiao.pan@kikatech.com
 */
public class CNNTestMain {

    public static void main(String[] args) throws IOException {
        ComputationGraph graph = ModelSerializer.restoreComputationGraph("./src/main/resources/model/model-highLearningRate04-40.txt");
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("glove.twitter.27B.100d.txt");
        EmojiToIndex emojiToIndex = new EmojiToIndex();
        INDArray happy = word2Vec.getWordVectorMatrix("happy");
//        INDArray zeros = Nd4j.zeros(1, 1, 30, 100);
        INDArray zeros = Nd4j.zeros(1, 1, 4, 100);
        zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()}, happy);

        INDArray birthday = word2Vec.getWordVectorMatrix("birthday");
        zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all()}, birthday);

        INDArray to = word2Vec.getWordVectorMatrix("to");
        zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all()}, to);

        INDArray you = word2Vec.getWordVectorMatrix("you");
        zeros.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(3), NDArrayIndex.all()}, you);
        INDArray[] output = graph.output(zeros);
        for (int i = 0; i < output.length; i++) {
            INDArray indArray = output[i];
            List<String> emojiTopN = new ArrayList<>();
            Preconditions.checkArgument(indArray.length() == emojiToIndex.totalOutputNumber());
            Map<String, Float> scoreMap = new HashMap<>();
            List<String> emojis = new ArrayList<>(emojiToIndex.getEmojiToIndexMap().keySet());
            for (int j = 0; j < indArray.length(); j++) {
                scoreMap.put(emojis.get(j), indArray.getFloat(j));
            }

            scoreMap.entrySet().stream().sorted(((o1, o2) -> -o1.getValue().compareTo(o2.getValue())))
                    .forEachOrdered(entry -> emojiTopN.add(entry.getKey()));
            System.out.println(emojiTopN);
        }
    }
}
