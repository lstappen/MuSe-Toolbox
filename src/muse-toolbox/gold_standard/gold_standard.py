from gold_standard.alignment import align
from gold_standard.fusion import fuse


def gold_standard(args):
    args.aligned = False

    # Print settings
    print(f'Alignment: {args.alignment}')
    print(f'Fusion: {args.fusion}')
    print(f'Std: {"standardize annos per video " if args.std_annos_per_sample else ""}'
          f'{"standardize annos over all videos " if args.std_annos_all_samples else ""}'
          f'{"no standardization " if not args.std_annos_per_sample and not args.std_annos_all_samples else ""}')
    print(f'Smoothing: pre-smoothing {args.pre_smoothing} '
          f'{"[window: " + str(args.pre_smoothing_window) + "] " if args.pre_smoothing != "none" else ""}'
          f' | post-smoothing {args.pre_smoothing_window}')

    if args.alignment == 'none' and args.fusion == 'none':
        print('Both alignment and fusion are set to none. Nothing happens.')

    # Alignment
    if args.alignment in ['ctw']:
        args.input_path = align(args)
        # set all preprocessing options to False/none to make sure preprocessing is only done once
        args.std_annos_per_sample, args.std_annos_all_samples, args.pre_smoothing = False, False, 'none'
        args.aligned = True
    elif args.alignment is not 'none':
        print("Alignment method not implemented. Please choose from ['ctw', 'none'].")

    # Fusion
    if args.fusion in ['mean', 'dba', 'ewe']:
        fuse(args)
    elif args.fusion is not 'none':
        print("Fusion method not implemented. Please choose from ['mean', 'dba', 'ewe', 'none'].")
