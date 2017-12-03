package me.peyppicp.advance;

import com.google.common.base.Preconditions;
import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class RemoveEmoji2 {

    public static void main(String[] args) throws IOException {
        String path = "F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\standardData.txt";
        String labelPath = "F:\\WorkSpace\\idea project location\\AI-Emoji\\label.txt";
        String path1 = "F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\distinctLines2.txt";
        String path2 = "F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\commonLabel2.txt";
        List<String> dataLines = FileUtils.readLines(new File(path), Charsets.UTF_8);
        List<String> emojiLines = FileUtils.readLines(new File(labelPath), Charsets.UTF_8);
        Map<String, String> map = new HashMap<>();
        ArrayList<Emoji> emojis = new ArrayList<>(EmojiManager.getAll());
        int threadNum = 10;

//       去除句尾的emoji
        for (int i = 0; i < dataLines.size(); i++) {
            String line = dataLines.get(i);
            StringBuilder emojiStr = new StringBuilder();
            boolean isChange = false;
            for (int j = 0; j < emojis.size(); j++) {
                if (line.length() >= 2 && line.substring(line.length() - 2, line.length()).equals(emojis.get(j).getUnicode())) {
                    try {
                        line = line.replaceAll(emojis.get(j).getUnicode(), "").trim();
                        isChange = true;
                        emojiStr.append(emojis.get(j).getUnicode()).append("\t");
                        j = 0;
                    } catch (Exception e) {
                        System.out.println(line);
                    }
                }
            }
            if (!StringUtils.isEmpty(line) && isChange) {
                map.putIfAbsent(line, emojiStr.toString());
                System.out.println("Pre:" + i + " has finished. Remain:" + (dataLines.size() - i) + ".");
            }
        }
        File file1 = new File(path1);
        File file2 = new File(path2);
        file1.createNewFile();
        file2.createNewFile();

        ArrayList<String> midKeys = new ArrayList<>(map.size());
        ArrayList<String> midValues = new ArrayList<>(map.size());
        map.forEach((s, s2) -> {
            midKeys.add(s);
            midValues.add(s2);
        });

        Preconditions.checkState(midKeys.size() == midValues.size());

        ArrayList<String> finalKeys = new ArrayList<>();
        ArrayList<String> finalValues = new ArrayList<>();

//        去除句中的emoji
        for (int i = 0; i < midKeys.size(); i++) {
            String line = midKeys.get(i);
            StringBuilder label = new StringBuilder(midValues.get(i));
            boolean hasEmoji = false;
            for (int j = 0; j < emojis.size(); j++) {
                try {
                    if (line.contains(emojis.get(j).getUnicode())) {
                        line = line.replaceAll(emojis.get(j).getUnicode(), "");
                        hasEmoji = true;
                        label.append(emojis.get(j).getUnicode());
                        j = 0;
                    }
                } catch (Exception e) { }
            }
            if (!hasEmoji) {
                finalKeys.add(line);
                finalValues.add(label.toString());
            }
            System.out.println("Mid:" + i + " has finished. Remain:" + (midKeys.size() - i) + ".");
        }

        ArrayList<String> postKeys = new ArrayList<>(finalKeys);
        ArrayList<String> postValues = new ArrayList<>(finalValues);

        ArrayList<String> trueKeys = new ArrayList<>();
        ArrayList<String> trueValues = new ArrayList<>();

        Preconditions.checkState(postKeys.size() == postValues.size());
        DefaultTokenizerFactory defaultTokenizerFactory = new DefaultTokenizerFactory();

        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("word2vecFull.txt");

//        去除句子长度为一个
        for (int i = 0; i < postKeys.size(); i++) {
            List<String> tokens = defaultTokenizerFactory.create(postKeys.get(i)).getTokens();
            if (tokens.size() >= 2) {
                trueKeys.add(postKeys.get(i));
                trueValues.add(postValues.get(i));
            } else if (tokens.size() == 1) {
                String s = tokens.get(0);
                Collection<String> strings = word2Vec.wordsNearest(s, 10);
                if (strings.size() > 0) {
                    trueKeys.add(postKeys.get(i));
                    trueValues.add(postValues.get(i));
                }
            }
            System.out.println("True:" + i + " has finished. Remain:" + (postKeys.size() - i) + ".");
        }

        FileUtils.writeLines(file1, Charsets.UTF_8.displayName(), trueKeys, false);
        FileUtils.writeLines(file2, Charsets.UTF_8.displayName(), trueValues, false);
    }
}
