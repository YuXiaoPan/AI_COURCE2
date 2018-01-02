package me.peyppicp.ai;

import me.peyppicp.Utils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/28
 * @email yuxiao.pan@kikatech.com
 */
public class PTBTestMain {

    public static void main(String[] args) throws IOException {
        String prefix = "";
        String output = "";
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("/Users/yuxiao.pan/IdeaProjects/AI_COURCE2/src/main/resources/model/more_words11.txt");
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(prefix + "glove.twitter.27B.50d.txt");
        WordToIndex wordToIndex = new WordToIndex(prefix + "pair.txt");
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        RnnTest rnnTest = new RnnTest(wordVectors, wordToIndex, tokenizerFactory, model, null, 3);
        List<String> collect = Utils.readLinesFromPath(prefix + "emoji_sample_head.txt")
                .stream().limit(50000).collect(Collectors.toList());
        Collections.shuffle(collect);
        for (String s : collect) {
            rnnTest.generateTokensFromStr(s, 10);
        }
//        int threadNum = 8;
//        List<List<String>> testDatas = new ArrayList<>();
//        for (int i = 0; i < threadNum; i++) {
//            testDatas.add(new ArrayList<>());
//        }
//        for (int i = 0; i < collect.size(); i++) {
//            int number = i % threadNum;
//            List<String> strings = testDatas.get(number);
//            strings.add(collect.get(i));
//        }
//
//        ExecutorService executorService = Executors.newFixedThreadPool(threadNum);
//        for (int i = 0; i < threadNum; i++) {
//            executorService.submit(new EvaluationRunner(ModelSerializer.restoreMultiLayerNetwork(prefix + "ptb2.txt"),
//                    wordVectors, new WordToIndex(prefix + "pair.txt"), testDatas.get(i), tokenizerFactory));
//        }
//
//        System.out.println("--------------");
//        System.out.println(PTBEvaluation.getInstance().getCorrectTop1Rate());
//        System.out.println(PTBEvaluation.getInstance().getCorrectTop3Rate());
    }
}

class EvaluationRunner implements Runnable {

    private MultiLayerNetwork multiLayerNetwork;
    private WordVectors wordVectors;
    private WordToIndex wordToIndex;
    private List<String> testData;
    private DefaultTokenizerFactory defaultTokenizerFactory;

    public EvaluationRunner(MultiLayerNetwork multiLayerNetwork, WordVectors wordVectors, WordToIndex wordToIndex, List<String> testData, DefaultTokenizerFactory defaultTokenizerFactory) {
        this.multiLayerNetwork = multiLayerNetwork;
        this.wordVectors = wordVectors;
        this.wordToIndex = wordToIndex;
        this.testData = testData;
        this.defaultTokenizerFactory = defaultTokenizerFactory;
    }

    @Override
    public void run() {
        RnnTest rnnTest = new RnnTest(wordVectors, wordToIndex, defaultTokenizerFactory, multiLayerNetwork, null, 3);
        for (String s : testData) {
            rnnTest.generateTokensFromStr(s, 10);
        }
    }
}
