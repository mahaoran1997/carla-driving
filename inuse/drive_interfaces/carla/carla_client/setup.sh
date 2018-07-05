
# Get protobuf source
if [[ ! -d "protobuf-source" ]]; then
  echo "Retrieving protobuf..."
  git clone --depth=1 -b v3.3.0  https://github.com/google/protobuf.git protobuf-source
else
  echo "Folder protobuf-source already exists, skipping git clone..."
fi

pushd protobuf-source >/dev/null

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/../llvm-install/lib/"

./autogen.sh
./configure \
    CC="clang-3.9" \
    CXX="clang++-3.9" \
    CXXFLAGS="-fPIC -stdlib=libc++ -I$PWD/../llvm-install/include/c++/v1" \
    LDFLAGS="-stdlib=libc++ -L$PWD/../llvm-install/lib/" \
    --prefix="$PWD/../protobuf-install"
make
make install

popd >/dev/null