package me.peyppicp.advance;

import com.vdurmont.emoji.EmojiManager;
import com.vdurmont.emoji.EmojiParser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class TestMain {

    public static void main(String[] args) throws IOException {

//        roof🐻
        List<String> temp = new ArrayList<>();
            try {
                String line = "roof\uD83D\uDC3B";
                int emojiLength = 2;
                int currentEmojiIndex = 0;
                List<String> containedEmojis = EmojiParser.extractEmojis(line).parallelStream().distinct().collect(Collectors.toList());
                boolean flag = false;
                for (String emoji : containedEmojis) {
                    currentEmojiIndex = line.indexOf(emoji);
                    if (currentEmojiIndex != -1) {
                        for (int i = currentEmojiIndex; i < line.length() - 1; i += emojiLength) {
                            if (EmojiManager.isEmoji(line.substring(currentEmojiIndex, currentEmojiIndex + emojiLength))) {
                                currentEmojiIndex += emojiLength;
                                flag = true;
                            }
                        }
                    }
                    if (flag) {
                        temp.add(line.substring(0, currentEmojiIndex).trim().toLowerCase());
                        line = line.substring(currentEmojiIndex, line.length()).trim().toLowerCase();
                    }
                }
                if (!flag) {
                    temp.add(line.toLowerCase());
                }
            } catch (Exception e) {
//                errorLines.add(line);
            }

//        temp = temp.parallelStream().filter(s -> EmojiParser.extractEmojis(s).size() != s.length() / 2).distinct().collect(Collectors.toList());
        List<String> temp1 = new ArrayList<>();

//        添加空格
        for (String sample : temp) {
            if (EmojiParser.extractEmojis(sample).size() > 0) {
                String emoji = EmojiParser.extractEmojis(sample).get(0);
                int i = sample.indexOf(emoji);
                if (i >= 1) {
                    if (i == emoji.length() - 1) {
                        continue;
                    } else {
                        String head = sample.substring(0, i);
                        String last = sample.substring(i, sample.length());
                        temp1.add(head + " " + last);
                    }
                }
            }else{
                temp1.add(sample);
            }
        }
        System.out.println();
    }
}
