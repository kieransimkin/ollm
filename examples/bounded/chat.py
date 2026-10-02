"""Local bounded text inference, including native DeepSeek without tool protocols."""
from common import parser,inference


def main():
    p=parser(__doc__)
    p.set_defaults(prompt='Explain expert streaming and its storage-speed trade-offs.')
    args=p.parse_args()
    o=inference(args)
    print(o.generate([{'role':'user','content':args.prompt}],max_new_tokens=o.budget.max_output_tokens))


if __name__=='__main__':
    main()
