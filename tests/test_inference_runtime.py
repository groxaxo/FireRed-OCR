import os
import unittest

from inference_runtime import (
    batched,
    build_worker_assignments,
    configure_single_gpu_process,
    generate_with_inference_mode,
    get_hf_model_load_kwargs,
    split_list,
)


class InferenceRuntimeTests(unittest.TestCase):
    def test_split_list_round_robins_items(self):
        self.assertEqual(split_list([1, 2, 3, 4, 5], 2), [[1, 3, 5], [2, 4]])

    def test_build_worker_assignments_skips_empty_chunks(self):
        assignments = build_worker_assignments(["a", "b"], 4)
        self.assertEqual(assignments, [(0, ["a"]), (1, ["b"])])

    def test_batched_groups_by_requested_batch_size(self):
        self.assertEqual(list(batched([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_configure_single_gpu_process_sets_visibility(self):
        previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        previous_order = os.environ.get("CUDA_DEVICE_ORDER")
        try:
            configure_single_gpu_process(3)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        finally:
            if previous_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
            if previous_order is None:
                os.environ.pop("CUDA_DEVICE_ORDER", None)
            else:
                os.environ["CUDA_DEVICE_ORDER"] = previous_order

    def test_hf_model_load_kwargs_pin_to_single_visible_gpu(self):
        self.assertEqual(
            get_hf_model_load_kwargs(),
            {"torch_dtype": "auto", "device_map": {"": "cuda:0"}},
        )

    def test_generate_with_inference_mode_wraps_model_generate(self):
        calls = []

        class FakeInferenceMode:
            def __enter__(self):
                calls.append("enter")

            def __exit__(self, exc_type, exc, tb):
                calls.append("exit")

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        class FakeModel:
            @staticmethod
            def generate(**kwargs):
                calls.append(("generate", kwargs))
                return "ok"

        result = generate_with_inference_mode(
            FakeTorch(),
            FakeModel(),
            max_new_tokens=128,
            input_ids=[1, 2, 3],
        )

        self.assertEqual(result, "ok")
        self.assertEqual(
            calls,
            [
                "enter",
                ("generate", {"input_ids": [1, 2, 3], "max_new_tokens": 128}),
                "exit",
            ],
        )


if __name__ == "__main__":
    unittest.main()
