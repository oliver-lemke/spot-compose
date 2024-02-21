import openai
from utils.environment import set_api_keys
from utils.recursive_config import Config


def main(config: Config):
    set_api_keys(config)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a robot designed to help in the household. I give "
                'you a message that will say something like this: "Pick up '
                'the keys and place them on the counter". You will respond '
                "with two words, the object to be picked up, and the place "
                "where it should be placed. In this situation the answer "
                'should be "keys, counter"',
            },
            {"role": "user", "content": "Pick up the bottle and put it onto the desk."},
        ],
    )
    print(response)
    content = response.choices[0].message["content"]
    print(content)


if __name__ == "__main__":
    main(Config())
