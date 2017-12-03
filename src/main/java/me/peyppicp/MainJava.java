package me.peyppicp;

import java.io.IOException;

/**
 * @author YuXiao Pan
 * @date 2017/11/18
 * @email yuxiao.pan@kikatech.com
 */
public class MainJava {

    public static void main(String[] args) throws IOException {
//        File file = new File("/Users/yuxiao.pan/IdeaProjects/AI-Emoji/src/main/resources/emoji_sample.txt");
//        List<String> strings = Files.readLines(file, Charsets.UTF_8);
//        Set<Character> characterSet = new HashSet<Character>();
//        for (String string : strings) {
//            char[] chars = string.toCharArray();
//            for (int i = 0; i < chars.length; i++) {
//                characterSet.add(chars[i]);
//            }
//        }
//        System.out.println(Joiner.on(",").join(characterSet));
        System.out.println(org.apache.commons.io.Charsets.UTF_8.displayName());
    }
}
