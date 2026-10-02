"""Qwen3-VL-2B bounded image chat; image files stay local."""
from common import parser,inference


def main():
    p=parser(__doc__)
    p.add_argument('--image',type=str,action='append',required=True,
                   help='Local image path; repeat for multiple images')
    p.set_defaults(model_key='qwen3-vl-2b-instruct',
                   prompt='Describe the image carefully and mention any readable text.')
    args=p.parse_args()
    model=inference(args)
    content=[{'type':'image','image':path} for path in args.image]
    content.append({'type':'text','text':args.prompt})
    print(model.generate([{'role':'user','content':content}],max_new_tokens=args.max_new_tokens))
    print(model.model.last_report)


if __name__=='__main__':
    main()
