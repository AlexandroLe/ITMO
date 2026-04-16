package com.example;

import org.junit.jupiter.api.ClassDescriptor;
import org.junit.jupiter.api.ClassOrderer;
import org.junit.jupiter.api.ClassOrdererContext;

import java.util.*;

public class CustomClassOrderer implements ClassOrderer {
    private static final List<String> CLASS_ORDER = Arrays.asList(
            "TC02_LoginInvalidPasswordTest",
            "TC03_EmptyLoginFieldsTest",
            "TC04_FeedPostsTest",
            "TC05_OpenPostTest",
            "TC07_LikeWithoutAuthTest",
            "TC09_CommentWithoutAuthTest",
            "TC10_SearchExistingContentTest",
            "TC11_SearchInvalidContentTest",
            "TC15_RegisterExistingLoginTest",
            "TC16_RegisterInvalidEmailTest",
            "TC17_RegisterEmptyTest",
            "TC18_RegisterExistingEmailTest",
            "TC01_LoginTest",
            "TC06a_LikeAndRemoveLikeTest",
            "TC06_LikeWithAuthTest",
            "TC08_AddCommentTest",
            "TC13_CreateEmptyTitlePostTest"
    );

    @Override
    public void orderClasses(ClassOrdererContext context) {
        @SuppressWarnings("unchecked")
        List<ClassDescriptor> descriptors = (List<ClassDescriptor>) context.getClassDescriptors();
        descriptors.sort((d1, d2) -> {
            String name1 = d1.getTestClass().getSimpleName();
            String name2 = d2.getTestClass().getSimpleName();
            int idx1 = CLASS_ORDER.indexOf(name1);
            int idx2 = CLASS_ORDER.indexOf(name2);
            if (idx1 == -1) idx1 = Integer.MAX_VALUE;
            if (idx2 == -1) idx2 = Integer.MAX_VALUE;
            return Integer.compare(idx1, idx2);
        });
    }
}