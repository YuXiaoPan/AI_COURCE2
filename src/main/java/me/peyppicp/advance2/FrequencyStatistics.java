package me.peyppicp.advance2;

import com.google.common.base.Joiner;
import com.vdurmont.emoji.EmojiParser;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author YuXiao Pan
 * @date 2017/12/8
 * @email yuxiao.pan@kikatech.com
 */
public class FrequencyStatistics {

    public static void main(String[] args) throws IOException {
        File file = new File("EmojiSample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);

        HashMap<String, Integer> emojiCounter = new HashMap<>();
        for (String sample : samples) {
            List<String> extractEmojis = EmojiParser.extractEmojis(sample);
            for (String extractEmoji : extractEmojis) {
                emojiCounter.merge(extractEmoji, 1, (o, n) -> o + n);
            }
        }
        Map<String, Integer> temp = new LinkedHashMap<>();
        emojiCounter.entrySet().parallelStream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .filter(en -> en.getValue() >= 1000)
                .forEachOrdered(entry -> temp.put(entry.getKey(), entry.getValue()));
        System.out.println(Joiner.on("\n").join(temp.entrySet()));

//        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(new File("word2vecLookUpTable.txt"));
//        DefaultTokenizerFactory defaultTokenizerFactory = new DefaultTokenizerFactory();
//        ArrayList<String> strings = new ArrayList<>();
//        for (String sample : samples) {
//            List<String> tokens = defaultTokenizerFactory.create(sample).getTokens();
//            boolean flag = false;
//            for (String token : tokens) {
//                if (!word2Vec.hasWord(token)) {
//                    flag = true;
//                }
//            }
//            if (!flag) {
//                strings.add(sample);
//            }
//        }
//
//        System.out.println(Joiner.on("\n").join(strings));
    }
}
