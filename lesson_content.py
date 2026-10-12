"""Lesson ordering and source labels for the single-path tutorial."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LessonStep:
    chapter: str
    chapter_title: str
    title: str


LESSON_STEPS = [
    LessonStep("CHAPTER 1", "First, what are we even doing?", "First, there are only TWO phases."),
    LessonStep("CHAPTER 1", "First, what are we even doing?", "Imagine we have two helpers looking at a cat picture."),
    LessonStep("CHAPTER 1", "First, what are we even doing?", "Who decides which helper to listen to?"),
    LessonStep("CHAPTER 1", "First, what are we even doing?", "We have the weights. Now what?"),
    LessonStep("CHAPTER 1", "First, what are we even doing?", "Our main model uses SOFT prediction."),
    LessonStep("CHAPTER 1", "First, what are we even doing?", "Before we learn, let's check one thing."),
    LessonStep("CHAPTER 2", "Now we start training", "Now imagine we're TRAINING."),
    LessonStep("CHAPTER 2", "Now we start training", "Let's see who gave the better answer."),
    LessonStep("CHAPTER 3", "The problem the 1991 paper wanted to solve", "What if we only check the final combined answer?"),
    LessonStep("CHAPTER 3", "The problem the 1991 paper wanted to solve", "What if two wrong experts cancel each other's mistakes?"),
    LessonStep("CHAPTER 3", "The problem the 1991 paper wanted to solve", "So what did the scientists change?"),
    LessonStep("CHAPTER 3", "The problem the 1991 paper wanted to solve", "What does random expert selection have to do with this error?"),
    LessonStep("CHAPTER 4", "Why the paper goes even further", "The authors weren't finished yet."),
    LessonStep("CHAPTER 4", "Why the paper goes even further", "Let's calculate the loss, one tiny piece at a time."),
    LessonStep("CHAPTER 4", "Why the paper goes even further", "Who deserves credit now that we know the answer?"),
    LessonStep("CHAPTER 4", "Why the paper goes even further", "Now comes the clever part."),
    LessonStep("CHAPTER 5", "Watch one real learning update", "Okay, but HOW does the model learn?"),
    LessonStep("CHAPTER 5", "Watch one real learning update", "Let's make our experts a little better."),
    LessonStep("CHAPTER 5", "Watch one real learning update", "Now teach the gate which expert to trust."),
    LessonStep("CHAPTER 5", "Watch one real learning update", "Did the model improve after learning?"),
    LessonStep("CHAPTER 5", "Watch one real learning update", "But why do we need more than one expert?"),
    LessonStep("CHAPTER 5", "Watch one real learning update", "You now understand the entire idea."),
]


assert len(LESSON_STEPS) == 22


PAPER_CITATION = (
    "Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). "
    '“Adaptive Mixtures of Local Experts.” Neural Computation, 3(1), 79–87.'
)
