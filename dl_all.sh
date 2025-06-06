#!/bin/bash

urls=(
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AGLIT2RCQOFJbtGTA8dRTJc/005A_07021_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AMUyKuNIi-54-0NgH5W3SH0/012D_07075_080504.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ACwM4Hmxn9IZAeuDFplJCFk/014A_07688_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AEaWqfSKTXQr04Yyf5-dW0I/021D_09150_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AKa6fNytLPtrRPo1wjudN14/022D_04826_121209.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AAAzvw9JZxrPp922aMN3pjo/032D_07536_111313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AHYuEi-LmsdCCMPHiSO1DO0/032D_09854_070505.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ACHD1edScWblzl55rYEz1sA/049A_07091_051113.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AOf84Vy-pLvnFoI0YD1MWhU/076D_09725_121107.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AKf8ViG50NaLAcLuu4a7pPo/079D_05011_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AA8zhb11sGU1c-4Ve9mxvxQ/079D_07694_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ANWhwjL6x45oLENojaARTyk/081A_12881_141414.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AF7uovX_qfqLx0WegOwIkDA/083D_12440_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AKoFN-Jtnj2oq8_xcaEA-yA/083D_12636_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AJghA_GcWms5IMl6AKdxOEw/086A_06208_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AEQ91v7ya0F9ctoTH15lllY/087D_07004_060904.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AJJjGLwg-mGEKaZ9zb1szf8/103A_07010_121313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AOOuZEhKGhMr9D9PtvkuXhM/106A_09090_000404.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AG54q8pJsW-51aR_0TsSXWI/109D_05390_141615.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AMpXcHpVhEMDxtpqhxVdt70/115D_04601_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AGiOwsdWmf_F-6khJvCog14/115D_04999_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AHQomOcRad7uh1BMsoKdhAk/124D_04854_171313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AAHX5xqCgwV6d-rzmF57yls/124D_05291_081406.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ALjNguzoI0vcgpdFao90yno/128D_09016_110500.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AA0P3LdxN-3Qqytlov6NSUc/142A_07606_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AE3YWE8iZVwzq9gm1_Id3QY/144A_SM_REUN_S6.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ALBqCWFC_d415VIQhMjNr34/146D_12613_231107.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AC2DC-VnJPkYxUK0u0FWP8M/151D_SM_REUN_S4.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ALpG8KNAPDuOR8r4Ny_Xha0/152D_08915_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AF-8yTQ8QvweL3nwijI5EmI/152D_09114_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AEz3o5Pdp6238zsLS373Vl4/155D_02484_191814.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AJRaclJuTTxwwRbYxTia35E/155D_02579_100800.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AHsXU6OJzc1xVnamCTb72Cs/156A_05796_080500.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AJmFEgBfgLpNfBb0wY3-lG0/162A_06192_060402.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ACUMu8DqYFjQH2HQXfxJsJE/164A_13146_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AGEalt2CZKPER5kTJb4nNXE/169D_00001_020800.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/APbraRxYIhBe8xdM5sK7ly8/174A_09133_131313.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	#'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AFz9bvI-bIq0gqjt6zX6k34/175D_12845_111312.zarr.tar?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ACtTlBt9xgebYmTWCFowe6E/annotations.json?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/AILqCCRST_VXZK8aDoh5Zq8/metadata.json?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
	'https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ADIxQFwjlgJHpLmcU6252i8/timeseries.json?rlkey=ery6e44t5u0osgryyt1qq9z4b&dl=1'
)

# Check if a directory argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/download_directory/"
    exit 1
fi

DATASET_DIR="$1"

for url in "${urls[@]}"; do
	echo "Downloading Hephaestus Zarr Minicubes: $url"

	IFS='?' read -r -a array <<< $(basename "$url")

	curl -o "{$DATASET_DIR}${array[0]}" -L "$url"

	if [ $? -eq 0 ]; then
		echo "Successfully downloaded: $(basename "$url")"

	else
		echo "Failed to download: $url"

	fi
	echo ""

done

echo "All Hephaestus Minicubes Downloaded"
