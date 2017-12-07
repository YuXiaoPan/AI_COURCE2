package me.peyppicp.advance;

import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import com.vdurmont.emoji.EmojiParser;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class TestMain {

    public static void main(String[] args) throws IOException {

        List<String> emojiUnicodes = EmojiManager.getAll().parallelStream().map(Emoji::getUnicode).collect(Collectors.toList());

//        I like that even more 😍💯 we did not just get away with that doe 😭
//        String line = "I like that even more \uD83D\uDE0D\uD83D\uDCAF we did not just get away with that doe \uD83D\uDE2D";
        String line = "She's My Women Crush Wednesday && Always Has Been \uD83D\uDE0D❤ I Live You Lil Mama \uD83D\uDE0D\uD83D\uDE4F❤\uD83D\uDC8D";


        int emojiLength = 2;
        int currentEmojiIndex = 0;
        List<String> containedEmojis = EmojiParser.extractEmojis(line).parallelStream().distinct().collect(Collectors.toList());
        for (String emoji : containedEmojis) {
            currentEmojiIndex = line.indexOf(emoji);
            boolean flag = false;
            if (currentEmojiIndex != -1) {
                for (int i = currentEmojiIndex; i < line.length() - 1; i += emojiLength) {
                    if (EmojiManager.isEmoji(line.substring(currentEmojiIndex, currentEmojiIndex + emojiLength))) {
                        currentEmojiIndex += emojiLength;
                        flag = true;
                    }
                }
            }
            if (flag) {
                System.out.println(line.substring(0, currentEmojiIndex).trim());
                line = line.substring(currentEmojiIndex, line.length()).trim();
            }
        }


    }
}
